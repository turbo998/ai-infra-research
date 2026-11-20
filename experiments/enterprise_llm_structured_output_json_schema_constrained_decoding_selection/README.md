# Enterprise LLM Structured Outputs, XGrammar FSM Constrained Decoding & Reasoning-First Schema Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-llm-structured-output-json-schema-constrained-decoding-selection-faq.md`](../../docs/components/enterprise-llm-structured-output-json-schema-constrained-decoding-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-llm-structured-output-json-schema-constrained-decoding-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_llm_structured_output_json_schema_constrained_decoding_selection/run.py
python experiments/enterprise_llm_structured_output_json_schema_constrained_decoding_selection/run.py --check
python -m unittest tests/test_enterprise_llm_structured_output_json_schema_constrained_decoding_selection.py
```
