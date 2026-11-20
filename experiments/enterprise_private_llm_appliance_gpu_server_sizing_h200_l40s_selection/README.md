# Enterprise On-Premises Private LLM Appliance & GPU Server Sizing (`R496`)

Deterministic quantitative evaluation model comparing four on-premises private AI server and appliance sizing architectures for **32K-context enterprise RAG & Copilot workloads (`Qwen2.5-72B / Llama-3.3-70B` + Embedding/Reranker/OCR)**:
- **Tier 1**: Unquantized `BF16` 70B/72B Model Across PCIe-Only `8x L40S (384GB)` without NVLink (`TP=8 over PCIe`)
- **Tier 2**: Oversized `8x H100 SXM5 (640GB)` Server + Mandatory Data-Center Power/Cooling Retrofit for Small 30-Concurrency Branch
- **Tier 3**: Dual `4x L40S PCIe` Nodes Running `AWQ INT4` 72B Without FP8 KV Cache or Embedding/Rerank GPU Isolation
- **Tier 4**: Right-Sized Tiered Appliance (`<=40 Users`: Air-Cooled `4x L40S FP8/AWQ + FP8 KV`; `50–250 Users`: `8x H200 1,128GB SXM` with `MIG` Partitioning for RAG/OCR)

## Verification

```bash
python experiments/enterprise_private_llm_appliance_gpu_server_sizing_h200_l40s_selection/run.py --check
python -m unittest tests/test_enterprise_private_llm_appliance_gpu_server_sizing_h200_l40s_selection.py
```
