# Operator-Level Selective Activation Rematerialization (`SAC`) & PCIe Gen5 Zero-Bubble Offloading

Deterministic CPU/offline model evaluating operator-level 0-1 knapsack value ratios ($\eta_{\text{op}} = \text{Bytes\_Saved} / \text{Recompute\_FLOPs}$) and NUMA-bound PCIe Gen5 dual-stream activation offloading for an 80-layer 70B GQA / MoE-w13 Transformer (`16,384` tokens/rank):

1. **`no_recompute_save_all`**: `467.81 GiB` HBM activation footprint (OOM on 80GB/141GB GPUs).
2. **`full_block_remat_checkpoint`**: Reduces HBM activations to `20.00 GiB` (`-95.72%`) at the cost of **`+33.33%`** step time overhead.
3. **`megatron_core_attn_pointwise_sac`**: Recomputes high-$\eta_{\text{op}}$ pointwise ops (`RMSNorm`, `RoPE`, `SwiGLU`, `DropoutAdd`) + `FlashAttention` core (`-56.18%` memory, `+2.43%` step time).
4. **`torchtitan_sac_recompute_w13_regions`**: Recomputes pointwise + `*routed_experts.w13.*` (`-81.76%` memory, `+16.97%` step time).
5. **`sac_knapsack_plus_pcie_gen5_offload`**: Combines knapsack SAC with NUMA-bound pinned PCIe Gen5 (`56 GB/s` effective DMA) dual-stream `saved_tensors_hooks` offloading, cutting HBM activation memory by **`86.94%`** (`467.81 -> 61.09 GiB`) with only **`+2.53%`** step time overhead (`10.61 ms` exposed PCIe stall vs `8,003.57 ms` on pageable non-NUMA Gen4).

## Run & Verify

```bash
python experiments/selective_activation_offload/run.py
python experiments/selective_activation_offload/run.py --check
```
