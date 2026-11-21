# Enterprise LLMOps Observability & Tracing Selection (`Azure AI Foundry Tracing` vs. `LangSmith` vs. `Arize Phoenix` vs. `Langfuse v3`)

Reproducible analytical model comparing four enterprise LLMOps observability, OpenTelemetry `gen_ai.*` tail-sampling, and online evaluation architectures at 50 million traces/month across `added latency`, `bad-case capture rate`, `storage footprint (TB)`, and `monthly TCO`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_llmops_observability_langfuse_langsmith_arize_foundry_tracing_tco_selection/run.py`
