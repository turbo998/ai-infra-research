# Ultra-Long Context (128K-1M) LLM Architecture: Hybrid Ulysses×Ring CP, Tiered KV Offload & Prefix Routing Benchmark

Deterministic quantitative evaluation model for [`docs/components/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-faq.md`](../../docs/components/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-faq.md) and [`.en.md`](../../docs/components/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/long_context_128k_1m_ring_ulysses_cp_kv_offload_prefix_routing_selection/run.py
python experiments/long_context_128k_1m_ring_ulysses_cp_kv_offload_prefix_routing_selection/run.py --check
python -m unittest tests/test_long_context_128k_1m_ring_ulysses_cp_kv_offload_prefix_routing_selection.py
```
