# LLM Inference Engine & PD Disaggregation Selection Sizer (`own-offline`)

Zero-dependency Python 3.12 analytical and systems model supporting:
- **[中文架构选型 FAQ (`docs/components/llm-inference-engine-pd-disaggregation-selection-faq.md`)](../../docs/components/llm-inference-engine-pd-disaggregation-selection-faq.md)**
- **[English Architecture Selection FAQ (`docs/components/llm-inference-engine-pd-disaggregation-selection-faq.en.md`)](../../docs/components/llm-inference-engine-pd-disaggregation-selection-faq.en.md)**
- **[7-Card 4:5 Benchmark Report (`docs/reports/llm-inference-engine-pd-selection-cards.html`)](../../docs/reports/llm-inference-engine-pd-selection-cards.html)**

## What This Experiment Verifies

1. **Roofline Physics (`evaluate_roofline_physics`)**:
   - Compares Llama-3-70B FP8 Prefill (`512.0 FLOP/B`, `2.49x` above H200 ridge point `205.5 FLOP/B`) vs. Decode at BS=32 (`31.2 FLOP/B`, `0.15x` of ridge point), demonstrating why Prefill is strictly compute-bound and Decode is strictly memory-bandwidth-bound.
2. **Multi-Turn Agent Prefix Reuse & Engine Optimizations (`evaluate_prefix_cache_and_engine_optimizations`)**:
   - Models SGLang `RadixAttention` on 16K agent contexts with 75% shared prefix + deferred unified page reuse (`#48852`, saving `2.4 ms` synchronous free stalls), slashing TTFT **`3.45x`** (`186.4 ms -> 54.0 ms`) and lifting QPS **`2.85x`**, alongside vLLM single-replay batch encoder CUDA graphs (`#65227`, **`3.20x`** launch reduction).
3. **Co-located Chunked Prefill vs. PD Disaggregation (`evaluate_colocated_vs_pd_disaggregation`)**:
   - Models mixed 32K RAG + interactive streaming chat. Co-located Chunked Prefill spikes Decode P99 TPOT to **`68.4 ms`** (violating `< 30 ms` SLA), while NVIDIA Dynamo + Mooncake 400G RDMA disaggregation isolates Decode P99 TPOT at **`18.2 ms`** (**`3.76x` lower tail latency**) and increases SLA-compliant goodput **`2.38x`** (`680.0 -> 1,618.4 tok/s/GPU`).
4. **Heterogeneous PD Right-Sizing TCO (`evaluate_heterogeneous_pd_tco`)**:
   - Quantifies **`34.2%` lower cost per 1M SLA-compliant tokens** (`$0.76 -> $0.50`) when pairing compute-dense Prefill nodes with bandwidth-rich Decode nodes.

## Reproduce & Verify

```bash
python experiments/llm_inference_engine_pd_selection/run.py
python experiments/llm_inference_engine_pd_selection/run.py --check
```
