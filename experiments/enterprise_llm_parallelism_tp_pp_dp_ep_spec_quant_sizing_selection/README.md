# Enterprise LLM Parallelism (`TP / PP / Attention DP / Expert EP`) & Acceleration Selection Sizer (`#509`)

Reproducible analytical model comparing **Tensor Parallelism (`TP`), Pipeline Parallelism (`PP`), Attention Data Parallelism (`DP`), and Expert Parallelism (`EP`)**, combined with **Speculative Decoding (`MTP / EAGLE-3`), Prefix Caching (`RadixAttention`), and Quantization (`Block-FP8` vs. `W4A16 AWQ`)** across **Dense (`70B / 405B`)** and **Sparse MoE (`DeepSeek 671B`)** models.

## Run

```bash
python experiments/enterprise_llm_parallelism_tp_pp_dp_ep_spec_quant_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-llm-parallelism-tp-pp-dp-ep-spec-quant-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-llm-parallelism-tp-pp-dp-ep-spec-quant-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/llm-parallelism-tp-pp-dp-ep-spec-quant-sizer-lab.html](../../docs/h5/llm-parallelism-tp-pp-dp-ep-spec-quant-sizer-lab.html)
