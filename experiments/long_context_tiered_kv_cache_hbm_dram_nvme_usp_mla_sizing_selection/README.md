# Experiment: 128K–1M Long-Context Tiered KV Cache Offload, 2D USP & MLA/Hybrid Attention Sizing (`#401`)

Reproducible analytical model comparing 128K–1M KV footprints (`GQA-8` vs. `DeepSeek MLA` vs. `7:1 Hybrid SSM/GQA`), multi-turn KV restore latency (`Pinned DRAM / RDMA / GDS NVMe` vs. `O(S^2) Prefill Recompute`), and `2D Context Parallelism (Ulysses × Ring)`.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/long_context_tiered_kv_cache_hbm_dram_nvme_usp_mla_sizing_selection/run.py`
- **Verify**: `python experiments/long_context_tiered_kv_cache_hbm_dram_nvme_usp_mla_sizing_selection/run.py --check`
