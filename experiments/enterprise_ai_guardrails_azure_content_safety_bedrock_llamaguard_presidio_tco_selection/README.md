# Enterprise AI Guardrails & PII Redaction Selection (`Azure AI Content Safety` vs. `AWS Bedrock Guardrails` vs. `Llama-Guard-3 + Microsoft Presidio`)

Reproducible analytical model comparing four enterprise AI safety guardrail and PII redaction architectures across `added TTFT latency`, `prompt injection block rate`, `PII leak rate`, and `monthly TCO` at 50 million requests/month.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_ai_guardrails_azure_content_safety_bedrock_llamaguard_presidio_tco_selection/run.py`
