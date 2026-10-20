# SGLang DSpark Simulated-Acceptance Fast Path, Radix-Cache Reclaimable Value Lengths, Sub-Pool End Allocation & vLLM ROCm Gated RMSNorm

## Overview / 概述

- **English**: Deterministic Python 3.12 standard-library verification model (`own-offline`) covering four upstream commits landed on 2026-10-20:
  1. `sgl-project/sglang@7d9bc10493` (`#45421`): Enabling DSpark simulated-acceptance fast paths (`accept_sampling_info = None`, `simulate_fold`, `_simulate_acc_greedy`) on NVIDIA CUDA instead of gating on `_is_hip`, saving `2 * B * K * V` (`24,330,240` for `B=16, K=5, V=152064`) discarded draft/target softmax evaluations per verify step.
  2. `sgl-project/sglang@0b536ca112` (`#45238`): Separating logical `value_len` from allocator-backed `reclaimable_value_len` (`device_reclaimable_value_len` in Rust) across `FULL`, `SWA`, and `MAMBA` tree components, plus guarding `adjust_component_size` against negative totals.
  3. `sgl-project/sglang@bab4a808d8` (`#45222`): Lazy ascending device sort (`_sorted_free_phys_pages`) and end-first allocation (`_take_free_pages_from_end` from `"low"` for `grow_direction=="up"` and `"high"` for `grow_direction=="down"`) in `MultiEndedAllocator`.
  4. `vllm-project/vllm@9da5f2d310` (`#60250`): Registering `torch.ops.vllm.fla_rms_norm_gated` (`direct_register_custom_op`) and enabling `+rms_norm_gated` by default on ROCm for unquantized models (e.g., Qwen3.8-27B).
- **中文**：针对 SGLang DSpark CUDA 模拟接受快路径、Radix-Cache 逻辑长度与可回收槽位长度解耦、Unified-Memory 双端池首尾优先分配，以及 vLLM ROCm `fla_rms_norm_gated` 自定义算子默认启用的确定性验证模型。

## Run / 运行方式

```bash
python experiments/sglang_dspark_sim_accept_radix_reclaimable_subpool_vllm_rocm/run.py --check
```
