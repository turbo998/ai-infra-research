# Enterprise AI Gateway Outbound `DLP`, Reversible Pseudonymization Vault (`FF1-FPE`) & Stream Re-Hydration (`R461`)

Deterministic evaluation suite comparing four enterprise AI Gateway outbound Data Loss Prevention (`DLP`), PII pseudonymization, and cross-border residency routing architectures across `4.5M` daily prompts (`18.5%` containing PII/secrets, `35.0%` involving MCP tool calls across `6` jurisdictions):

1. **Tier 1**: Unfiltered Pass-Through Proxy + Contractual ZDR Only (`14.6%` cloud PII/secret exposure, `11.4%` cross-border residency violation, `$195,000/mo` risk/rework cost)
2. **Tier 2**: Destructive Static `<REDACTED>` / `***` Regex Masking Gateway (`3.8%` PII exposure, `31.4%` coreference & MCP tool-call failure, `$148,000/mo` cost)
3. **Tier 3**: Heavy 7B LLM Inline Scrubber + Full-Response Buffering (`0.6%` exposure, `6.8%` tool error, `640.0 ms` added P99 TTFT latency, `$112,000/mo` cost)
4. **Tier 4**: `DeBERTa-ONNX` + `NIST FF1-FPE` Type-Preserving Vault + `Aho-Corasick` Streaming Re-Hydrator + Geo-Residency Router (`0.0%` cloud PII exposure, `0.4%` coreference/tool error, `14.0 ms` added P99 latency, `0.0%` residency violation, `$38,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection/run.py
python experiments/enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection/run.py --check
python -m unittest tests/test_enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection.py
```
