# Milestone `#515`: Enterprise 3-Tier KV Cache Pool (`HBM <-> Host Pinned DRAM <-> RDMA/3FS NVMe`) — `LMCache` vs. `Mooncake Store` vs. `NVIDIA Dynamo NIXL` Sizer

Reproducible analytical model evaluating **3-Tier Hierarchical KV Cache Offloading** for multi-turn agentic workflows (`500` concurrent `64K-context` agents pausing for tool execution), comparing `HBM-Only Recompute` (`1,840 ms` TTFT, `36 GPUs`) against `Local Host DRAM Offload` and `Global 3-Tier RDMA DRAM + 3FS NVMe GDS` (`118 ms` TTFT, `12 GPUs`, `-66.7%` GPUs).

## Run

```bash
python experiments/enterprise_milestone_515_tiered_kv_cache_lmcache_mooncake_nixl_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-milestone-515-tiered-kv-cache-lmcache-mooncake-nixl-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-milestone-515-tiered-kv-cache-lmcache-mooncake-nixl-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/milestone-515-tiered-kv-cache-lmcache-mooncake-nixl-lab.html](../../docs/h5/milestone-515-tiered-kv-cache-lmcache-mooncake-nixl-lab.html)
