# Enterprise 671B/235B MoE Serving & Expert Parallelism (TP vs. EP + EPLB) Evaluator 

Deterministic analytical evaluator comparing four enterprise deployment and parallelism architectures for serving a `671B-FP8` Mixture-of-Experts (`37B` active/token, `256` routed experts) model at a `15,000 tok/s` cluster target:

1. **Tier 1 (`tier1_naive_dual_node_tp16`)**: Naive Dual-Node Tensor Parallelism (`2x 8xH100 80GB TP=16`, `148 tok/s/GPU`, `82.0 ms` P99 TPOT due to 122 cross-node AllReduce syncs/token, `$96,000/mo`).
2. **Tier 2 (`tier2_single_node_8xh200_intranode_nvlink`)**: Single-Node `8x H200 (1,128 GB HBM3e)` Intra-Node `TP=8/EP=8` over 900 GB/s NVLink (`606 tok/s/GPU`, `23.5 ms` P99 TPOT, `$54,000/mo` — best turnkey architecture for `<= 24 GPUs` with zero cross-node RDMA tuning).
3. **Tier 3 (`tier3_multinode_static_ep16_no_eplb`)**: Multi-Node Static `EP=16` without Hot-Expert Load Balancing (`465 tok/s/GPU`, `3.40x` hot-expert straggler load imbalance ratio, `36.0 ms` P99 TPOT, `$64,000/mo`).
4. **Tier 4 (`tier4_wide_ep_deepep_overlap_plus_eplb`)**: Wide Expert Parallelism (`EP=16/32`) + Dual-Microbatch Compute-Communication Overlap (`DeepEP`) + Dynamic Redundant Hot-Expert Load Balancer (`EPLB`, `738 tok/s/GPU` [`4.99x` vs. Tier 1, `1.59x` vs. Tier 3], `1.08x` balanced expert load ratio, `19.5 ms` P99 TPOT, `$38,000/mo` [`-60.4%` vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_moe_serving_expert_parallelism_tp_vs_ep_selection/run.py
python experiments/enterprise_moe_serving_expert_parallelism_tp_vs_ep_selection/run.py --check
```
