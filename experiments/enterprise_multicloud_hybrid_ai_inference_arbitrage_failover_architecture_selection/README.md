# Experiment: Enterprise Multi-Cloud & Hybrid AI Inference Reference Architecture (`#413`)

Reproducible analytical model comparing `Single-Cloud Lock-In`, `Naive Cross-Cloud Per-Turn Round-Robin`, `Whole-Session Sticky Hybrid Burst`, and `Multi-Silicon vLLM (H200/MI325X/TPU) + Spot Arbitrage Pool` across prefix KV cache hit rate, cross-cloud outage resilience, and annual TCO.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_multicloud_hybrid_ai_inference_arbitrage_failover_architecture_selection/run.py`
- **Verify**: `python experiments/enterprise_multicloud_hybrid_ai_inference_arbitrage_failover_architecture_selection/run.py --check`
