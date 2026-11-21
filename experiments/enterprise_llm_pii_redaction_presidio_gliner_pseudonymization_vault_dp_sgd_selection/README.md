# Enterprise LLM PII Redaction & Reversible Pseudonymization Vault Experiment (`Presidio + GLiNER` vs. `Destructive Masking` vs. `DP-SGD`)

Reproducible analytical model comparing 4 enterprise PII/PHI privacy architectures across `2,048-token` multi-entity contracts (`18` PII entities/doc):
- **Tier 1 (`Naive Regex Destructive '[REDACTED]' Masking`)**: `66.0%` PII recall, `61.5%` downstream LLM task utility (destroys entity coreference), `+2.5 ms` latency.
- **Tier 2 (`Synchronous 8B LLM Scrubber`)**: `95.2%` PII recall, `+420.0 ms` blocking latency.
- **Tier 3 (`Differential Privacy SFT DP-SGD` $\epsilon=3.0$)**: `88.2%` utility (`-6.8 pts` perplexity penalty), offline training only.
- **Tier 4 (`Reversible Pseudonymization Vault`: `Microsoft Presidio` + `GLiNER ONNX` + `Bi-Directional `<PERSON_1>` Mapping & Streaming Restore`, Recommended)**: **`98.6%` PII recall**, **`94.8%` downstream task utility**, and **`+11.5 ms` gateway latency**.

## Run & Verify

```bash
python experiments/enterprise_llm_pii_redaction_presidio_gliner_pseudonymization_vault_dp_sgd_selection/run.py
python experiments/enterprise_llm_pii_redaction_presidio_gliner_pseudonymization_vault_dp_sgd_selection/run.py --check
```
