# 2D Hybrid Context Parallelism (`USP = Ulysses × ZigZag Ring-Attention`)

Deterministic CPU/offline model comparing 4 Context Parallelism (`CP = 64`, `8 nodes × 8 H100 SXM5 GPUs`) strategies across `256K–4M` sequence lengths for GQA (`H_q=64, H_kv=8, d=128`):

1. **`ulysses_64_gqa_replicated`**: Pure DeepSpeed-Ulysses across `64` GPUs (`8x` KV head replication due to `SP_ulysses > H_kv = 8` + cross-node RoCE All-to-All).
2. **`naive_ring_64`**: Contiguous 1D Ring-Attention across `64` GPUs (`49.22%` causal lower-triangle idle bubble + `63` fine-grained RoCE P2P ring hops).
3. **`usp_contiguous_8x8`**: 2D Hybrid `Ulysses=8` (intra-node NVLink) × `Contiguous-Ring=8` (inter-node RoCE), still subject to `43.75%` causal Ring imbalance.
4. **`usp_zigzag_8x8`**: 2D Hybrid `Ulysses=8` × `ZigZag-Ring=8` (`[r, 2R - 1 - r]` symmetric dual-chunk assignment), reducing causal imbalance to `0.16%` and achieving **`2.83x` (`1M`) to `2.82x` (`4M`)** throughput over naive Ring-Attention (`99.51%–99.76%` attention efficiency).

## Run & Verify

```bash
python experiments/context_parallel_usp_zigzag/run.py
python experiments/context_parallel_usp_zigzag/run.py --check
```
