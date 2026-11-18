# Enterprise Colocated vs. Prefill-Decode (PD) Disaggregated Inference Evaluator 

Deterministic analytical evaluator comparing four enterprise LLM serving architectures for an 8K-input / 512-output 70B RAG workload under strict dual SLA (`P95 TTFT <= 800 ms` AND `P99 TPOT <= 35 ms`):

1. **Tier 1 (`tier1_naive_colocated_continuous_batching`)**: Naive Colocated Continuous Batching (`1,150 tok/s` SLA Goodput, `620 ms` P95 TTFT, `148.0 ms` P99 TPOT severe stutter, `$92,000/mo` for 5K tok/s target).
2. **Tier 2 (`tier2_colocated_chunked_prefill`)**: Colocated + Tuned Chunked Prefill (`2,480 tok/s` SLA Goodput, `740 ms` P95 TTFT, `46.0 ms` P99 TPOT, `$46,000/mo` — best simple baseline for `<= 8 GPUs` without RDMA).
3. **Tier 3 (`tier3_pd_disagg_over_standard_tcp`)**: PD Disaggregation over Standard 25GbE TCP (`1,820 tok/s` SLA Goodput, `1,050 ms` P95 TTFT due to KV transfer bottleneck — anti-pattern).
4. **Tier 4 (`tier4_rdma_heterogeneous_pd_disaggregation`)**: RDMA-Backed Heterogeneous PD Disaggregation (`10P:6D` ratio over 200G/400G RoCEv2/IB + FP8 KV Transfer, `5,420 tok/s` SLA Goodput [`2.19x` vs. Tier 2, `4.71x` vs. Tier 1], `490 ms` P95 TTFT, `21.5 ms` P99 TPOT, `$25,000/mo` [`-45.7%` vs. Tier 2]).

## Usage

```bash
python experiments/enterprise_pd_disaggregation_vs_colocated_inference_selection/run.py
python experiments/enterprise_pd_disaggregation_vs_colocated_inference_selection/run.py --check
```
