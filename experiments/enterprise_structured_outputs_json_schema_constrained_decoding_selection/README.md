# Enterprise Structured Outputs & JSON Schema Constrained Decoding Evaluator 

Deterministic analytical evaluator comparing four enterprise structured output and JSON Schema enforcement architectures across `20,000,000` microservice API calls/month:

1. **Tier 1 (`tier1_prompt_only_regex_retry`)**: Prompt-Only Soft JSON Instruction + Regex Strip + Retry Loop (`90.6%` first-pass JSON syntax validity, `81.2%` semantic accuracy, `4,850 ms` P99 latency due to retries, `$18,600/mo`).
2. **Tier 2 (`tier2_legacy_cpu_cfg_masking`)**: Legacy Uncached CPU Interpreter CFG Masking (`100.0%` syntax validity, `84.5%` semantic accuracy, `+28.0 ms/token` CPU mask bottleneck, `3,420 ms` P99 latency, `$24,200/mo`).
3. **Tier 3 (`tier3_xgrammar_llguidance_naive_schema`)**: Pre-Compiled Zero-Overhead Constrained Decoding (`XGrammar / llguidance` with naive field order where `decision` precedes `reasoning`, `100.0%` syntax validity, `84.5%` semantic accuracy, `0.04 ms/token` overhead, `1,180 ms` P99 latency, `$11,200/mo`).
4. **Tier 4 (`tier4_reasoning_first_schema_xgrammar_pydantic_gate`)**: Reasoning-First Field Ordering (`rationale_summary` before `decision_enum`) + Flattened `XGrammar / llguidance` + Deterministic Pydantic `@model_validator` Gate (`100.0%` syntax validity, `97.8%` semantic accuracy [`+13.3 pts` over Tier 3], `0.03 ms/token` overhead, `960 ms` P99 latency [`5.05x` faster than Tier 1], `$9,800/mo` [`-47.3%` vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_structured_outputs_json_schema_constrained_decoding_selection/run.py
python experiments/enterprise_structured_outputs_json_schema_constrained_decoding_selection/run.py --check
```
