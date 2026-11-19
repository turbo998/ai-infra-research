# Enterprise LLM Structured Outputs, Constrained Decoding & Large-Scale Tool-Calling Selection (`R487`)

Deterministic evaluation suite comparing four enterprise LLM structured JSON generation and `250-tool` `MCP` function-calling architectures across `40,000,000` monthly calls (`250` registered enterprise tools = `45,000` raw schema tokens):

1. **Tier 1**: Prompt-Only `'Return Valid JSON'` + Regex Repair + Dumping All 250 Tools (`81.4%` first-pass JSON validity, `54.5%` tool accuracy, `1,950ms` P95 latency, `$164,000/mo` TCO)
2. **Tier 2**: Unconstrained Generation + Client Retry Loop (`Instructor / Pydantic` 3x Retries + All 250 Tools) (`84.0%` first-pass validity, `58.2%` tool accuracy, `3,420ms` P95 latency, `$138,000/mo` TCO)
3. **Tier 3**: Naive Un-Cached FSM Constrained Decoding + All 250 Tools (`100.0%` JSON validity, `61.0%` tool accuracy, `1,480ms` P95 latency, `$89,000/mo` TCO)
4. **Tier 4**: Pre-Compiled Grammar-Constrained Decoding (`XGrammar / Outlines / Cloud Strict Schema`) + `Top-8` Semantic Tool Router (`45K -> 1.8K tokens`) + `Reasoning-First` Schema Field Ordering (`100.0%` first-pass JSON validity, `96.8%` tool selection accuracy, `380ms` P95 latency, `$27,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_structured_output_json_schema_constrained_decoding_selection/run.py
python experiments/enterprise_structured_output_json_schema_constrained_decoding_selection/run.py --check
python -m unittest tests/test_enterprise_structured_output_json_schema_constrained_decoding_selection.py
```
