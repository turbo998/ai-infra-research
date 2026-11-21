# First-Principles LLM Capacity Sizing (Roofline / 6ND / KV Cache), XProf/Nsight Profiling & Goodput Triage Benchmark

Deterministic quantitative evaluation model for [`docs/components/llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-selection-faq.md`](../../docs/components/llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-selection-faq.md) and [`.en.md`](../../docs/components/llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_first_principles_roofline_6nd_kv_xprof_goodput_triage_selection/run.py
python experiments/llm_first_principles_roofline_6nd_kv_xprof_goodput_triage_selection/run.py --check
python -m unittest tests/test_llm_first_principles_roofline_6nd_kv_xprof_goodput_triage_selection.py
```
