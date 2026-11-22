# Enterprise Tiered KV Cache Offload (`L1 HBM -> L2 RDMA Host DRAM -> L3 GDS NVMe`) vs. $O(S^2)$ Prefill Recompute Sizing Model (`#453`)

Reproducible analytical model comparing HBM-only LRU eviction + full $O(S^2)$ recomputation, single-node Host Pinned DRAM offload, distributed RDMA Host-DRAM pooling (`Mooncake / LMCache`), and 3-Tier Hierarchical KV Pooling with layer-wise pipelined prefetch (`Dynamo NIXL / Mooncake + 3FS GDS`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_tiered_kv_cache_offload_dram_nvme_gds_recompute_sizing_selection/run.py --check`
