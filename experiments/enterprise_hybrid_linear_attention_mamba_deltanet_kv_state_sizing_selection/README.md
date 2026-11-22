# Enterprise Hybrid Linear-Attention / Mamba-Transformer (`Qwen3-Next / MiniMax-01 / Jamba / Nemotron-H` Class) $O(1)$ Recurrent State vs. KV Cache Sizing (`#478`)

Reproducible analytical model comparing Pure Full-Softmax GQA-8 (`40.96 GB` per `256K` request), Pure Linear/Mamba-2 (`64%` needle recall), Sliding-Window Attention, and `7:1` Hybrid Linear-Full Attention (`Gated DeltaNet / Mamba-2 + 12.5% Full GQA` = `5.26 GB` per `256K` request, `99.1%` needle recall, `7.75x` higher concurrency).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_hybrid_linear_attention_mamba_deltanet_kv_state_sizing_selection/run.py --check`
