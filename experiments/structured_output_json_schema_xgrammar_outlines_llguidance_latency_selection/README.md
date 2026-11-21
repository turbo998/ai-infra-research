# Structured Output & Constrained Decoding Selection Experiment (`XGrammar` vs. `llguidance` vs. `Outlines` vs. Unconstrained Retry)

Reproducible parametric model comparing 4 structured output architectures across grammar compilation cold `TTFT` overhead, per-token CPU mask latency, first-pass JSON Schema validity, jump-forward token savings, and monthly cluster TCO.

## Run
```bash
python experiments/structured_output_json_schema_xgrammar_outlines_llguidance_latency_selection/run.py
python experiments/structured_output_json_schema_xgrammar_outlines_llguidance_latency_selection/run.py --check
```
