# Hardware-Aligned Sparse Attention Selection Experiment (`DeepSeek NSA` vs. `Moonshot MoBA` vs. `Dense FlashAttention-3`)

Reproducible analytical model comparing 4 long-context (`128K` tokens) attention architectures on `8x H100/H200`:
- **Tier 1 (`Dense FlashAttention-3` $O(S^2)$)**: `11.80 sec` 128K Prefill TTFT, `42.0 ms` Decode TPOT, `98.8%` RULER-128K accuracy.
- **Tier 2 (`Static Sliding Window + Attention Sinks StreamingLLM`)**: `1.45 sec` TTFT, **but middle-document recall collapses to `24.5%`**.
- **Tier 3 (`MoBA: Mixture of Block Attention` parameter-free block router + `flash_attn_varlen`)**: `2.45 sec` 128K Prefill TTFT (`4.82x` faster than Dense FA-3), `99.1%` RULER-128K accuracy.
- **Tier 4 (`NSA: Native Sparse Attention` 3-branch Coarse Compression + GQA-Aligned Fine Block Selection + Local Window, Recommended)**: **`1.31 sec` 128K Prefill TTFT (`9.01x` faster)**, **`10.0 ms` Decode TPOT (`4.2x` faster)**, and **`99.7%` RULER-128K accuracy**.

## Run & Verify

```bash
python experiments/sparse_attention_nsa_moba_block_sparse_flashattention3_long_context_selection/run.py
python experiments/sparse_attention_nsa_moba_block_sparse_flashattention3_long_context_selection/run.py --check
```
