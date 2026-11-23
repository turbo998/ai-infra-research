# Enterprise 100% Structured Output (`XGrammar / llguidance + Jump-Forward`) Selection Sizer (`#518`)

Reproducible analytical model comparing **Unconstrained Prompt Retry Loops (`6,420ms` P99 E2E, `86.8%` 10-step agent success)**, **Legacy Synchronous Python FSM Masking (`11.8ms/tok` CPU overhead)**, and **Zero-Overhead `XGrammar / llguidance` Pushdown Automaton Bitmasking (`0.04ms/tok`) + Jump-Forward Deterministic Key Splicing (`1,120ms` P99 E2E, `100.0%` compliance, `-50.0%` GPUs)**.

## Run

```bash
python experiments/enterprise_structured_output_xgrammar_llguidance_constrained_decoding_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-structured-output-xgrammar-llguidance-constrained-decoding-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-structured-output-xgrammar-llguidance-constrained-decoding-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/structured-output-xgrammar-llguidance-constrained-decoding-lab.html](../../docs/h5/structured-output-xgrammar-llguidance-constrained-decoding-lab.html)
