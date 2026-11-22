# Enterprise Long-Context (`128K–1M Tokens`) 2D Unified Sequence Parallelism (`2D USP: Intra-Node Ulysses × Inter-Node Ring`) & `TTFT` Sizing Model (`#461`)

Reproducible analytical model comparing Pure `TP=8`, Pure `Ring Attention`, Pure `DeepSpeed Ulysses`, and `2D Hybrid USP (Intra-Node NVLink Ulysses=8 × Inter-Node RDMA Ring=2 + FlashAttention-3 FP8)`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_long_context_2d_usp_ulysses_ring_attention_ttft_sizing_selection/run.py --check`
