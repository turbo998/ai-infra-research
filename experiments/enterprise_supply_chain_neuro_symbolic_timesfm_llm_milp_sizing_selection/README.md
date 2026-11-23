# Enterprise Supply Chain Control Tower (`5M SKUs`: `TimesFM/Chronos + LLM + MILP Solver`) Sizer (`#522`)

Reproducible analytical model comparing **Pure LLM Inventory Guessing (`61.5%` constraint satisfaction, `64 GPUs` ✕)**, **Legacy Static ERP Min-Max Rules (`6.4%` stockout rate)**, and the **Neuro-Symbolic Triad (`Chronos/TimesFM` Time-Series Foundation Model + `70B LLM` Natural-Language Disruption/Constraint Compiler + `HiGHS/Gurobi MILP` Mathematical Solver, `100.0%` constraint satisfaction, `1.9%` stockout rate, `8x H200 GPUs`, `-87.5%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_supply_chain_neuro_symbolic_timesfm_llm_milp_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-supply-chain-neuro-symbolic-timesfm-llm-milp-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-supply-chain-neuro-symbolic-timesfm-llm-milp-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/supply-chain-neuro-symbolic-timesfm-llm-milp-lab.html](../../docs/h5/supply-chain-neuro-symbolic-timesfm-llm-milp-lab.html)
