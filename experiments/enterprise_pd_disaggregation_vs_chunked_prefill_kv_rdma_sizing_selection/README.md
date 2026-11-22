# Enterprise PD Disaggregation vs. Colocated Chunked Prefill & KV RDMA Sizing Model (`#444`)

Reproducible analytical model comparing Colocated Continuous Batching, Colocated Chunked Prefill (`chunk_size=1024`), 25GbE TCP PD Disaggregation (anti-pattern), and Elastic RDMA PD Disaggregation (`NVIDIA Dynamo NIXL / Mooncake / SGLang PD Router / vLLM V1 + LMCache`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_pd_disaggregation_vs_chunked_prefill_kv_rdma_sizing_selection/run.py --check`
