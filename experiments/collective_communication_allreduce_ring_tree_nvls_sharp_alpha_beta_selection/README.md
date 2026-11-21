# AI Cluster Collective Communication (All-Reduce/AG/RS/A2A): Ring vs. Tree vs. NVLS/SHARPv3 Alpha-Beta Benchmark

Deterministic quantitative evaluation model for [`docs/components/collective-communication-allreduce-ring-tree-nvls-sharp-alpha-beta-selection-faq.md`](../../docs/components/collective-communication-allreduce-ring-tree-nvls-sharp-alpha-beta-selection-faq.md) and [`.en.md`](../../docs/components/collective-communication-allreduce-ring-tree-nvls-sharp-alpha-beta-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/collective_communication_allreduce_ring_tree_nvls_sharp_alpha_beta_selection/run.py
python experiments/collective_communication_allreduce_ring_tree_nvls_sharp_alpha_beta_selection/run.py --check
python -m unittest tests/test_collective_communication_allreduce_ring_tree_nvls_sharp_alpha_beta_selection.py
```
