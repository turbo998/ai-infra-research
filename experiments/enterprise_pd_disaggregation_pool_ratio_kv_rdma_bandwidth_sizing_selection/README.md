# Enterprise PD Disaggregation `P:D` Pool Ratio & KV Cache RDMA Bandwidth Sizer (`#508`)

Reproducible analytical model evaluating **Prefill-Decode (PD) Disaggregation** pool sizing (`1P:1D`, `3P:1D`, `1P:4D`, `1P:2D`) versus colocated chunked prefill, alongside **KV Cache RDMA transfer bandwidth (`GQA 160 KB/tok` vs. `MLA 34.3 KB/tok`)** and **NVIDIA Dynamo NIXL / Mooncake layer-wise asynchronous pipelining (`94.5%` overlap)**.

## Run

```bash
python experiments/enterprise_pd_disaggregation_pool_ratio_kv_rdma_bandwidth_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-pd-disaggregation-pool-ratio-kv-rdma-bandwidth-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-pd-disaggregation-pool-ratio-kv-rdma-bandwidth-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/pd-disaggregation-pool-ratio-kv-rdma-bandwidth-sizer-lab.html](../../docs/h5/pd-disaggregation-pool-ratio-kv-rdma-bandwidth-sizer-lab.html)
