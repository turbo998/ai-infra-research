# MoE Systems Architecture (DeepEP Normal/IBGDA, DualPipe/Dual-Batch Overlap & EPLB Load Balancing) Benchmark

Deterministic quantitative evaluation model for [`docs/components/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-faq.md`](../../docs/components/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-faq.md) and [`.en.md`](../../docs/components/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/moe_architecture_deepep_ibgda_dualpipe_eplb_load_balancing_selection/run.py
python experiments/moe_architecture_deepep_ibgda_dualpipe_eplb_load_balancing_selection/run.py --check
python -m unittest tests/test_moe_architecture_deepep_ibgda_dualpipe_eplb_load_balancing_selection.py
```
