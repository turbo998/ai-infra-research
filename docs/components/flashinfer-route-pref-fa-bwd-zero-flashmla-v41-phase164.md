# FlashInfer MoE Finalize+AllReduce 路由索引预取与确定性包命名、FlashAttention CuTe 空 Q/K 零梯度保护及 FlashMLA DeepSeek-V4.1 256-Splits 扩展（Phase 164 · Priority 173）

> **English Version**: [`flashinfer-route-pref-fa-bwd-zero-flashmla-v41-phase164.en.md`](flashinfer-route-pref-fa-bwd-zero-flashmla-v41-phase164.en.md)

## 1. 一手源码证据与分类表

| 证据编号 | 仓库与提交 | 发布日期 | 证据层级 | 核心改动路径 |
| :--- | :--- | :--- | :--- | :--- |
| `src-flashinfer-moe-finalize-allreduce-route-prefetch-2026` | `flashinfer-ai/flashinfer` `#8073` (`795601c070`) | 2026-10-31 | `implementation` | `csrc/experimental/moe_finalize_allreduce_fusion/*`: 在 `cake_moe_finalize_allreduce_fusion` 中将首 Token 路由索引（Route Indices）提前至控制字（Control Words）之前发起读取，降低 75% 控制字等待停顿 |
| `src-flashinfer-revert-sm120-nvfp4-longest-first-2026` | `flashinfer-ai/flashinfer` `#8075` (`5f1eae4ddc`) | 2026-10-31 | `implementation` | `flashinfer/experimental/cake_minimax_h3/*`: 回滚 SM120 NVFP4 变长注意力两类重叠计划的 `longest-first` 排空顺序改动（`#8040`），恢复尾延迟稳定性 |
| `src-flashinfer-dsa-indexer-det-names-2026` | `flashinfer-ai/flashinfer` `#8061` (`7078416325`) | 2026-10-31 | `implementation` | `csrc/experimental/dsa_indexer_det/*`: 为 DSA Indexer Top-K 编译包引入确定性符号命名，消除多进程 JIT 缓存哈希漂移 |
| `src-flashinfer-rmsnorm-silu-autograd-det-names-2026` | `flashinfer-ai/flashinfer` `#8045` (`c677b26a0f`) | 2026-10-31 | `implementation` | `csrc/experimental/rmsnorm/*`: 为 RMSNorm 与 SiLU-Gated RMSNorm Autograd 编译包引入确定性命名 |
| `src-flashattn-cute-bwd-empty-qk-zero-grad-2026` | `Dao-AILab/flash-attention` `#2776` (`008e63667e`) | 2026-09-14 | `implementation` | `flash_attn/cute/flash_bwd.py`: 在 CuTe Backward 路径上为空 Q/K 负载显式返回全零梯度，消除 256 个未初始化 NaN 梯度元素 |
| `src-cutlass-fmha-regression-quick-fix-2026` | `NVIDIA/cutlass` `#3387` (`d01487db82`) | 2026-07-16 | `implementation` | `examples/41_fused_multi_head_attention/*`: 修复 CUTLASS FMHA 回归缺陷 |
| `src-triton-cd-free-threaded-314t-315t-wheels-2026` | `triton-lang/triton` `#12502` (`8fa612157a`) | 2026-10-26 | `implementation` | `.github/workflows/*`: 构建无 GIL（Free-Threaded）Python `3.14t` 与 `3.15t` 预编译 Wheel |
| `src-triton-amd-new-codegen-build-test-2026` | `triton-lang/triton` `#12391` (`b10226036e`) | 2026-10-21 | `implementation` | `third_party/amd/*`: 切换至新 AMD CodeGen LLVM 构建目标并更新回归测试 |
| `src-flashmla-st-async-longlong2-msvc-llp64-2026` | `deepseek-ai/FlashMLA` `#224` (`ba89a3466e`) | 2026-09-15 | `implementation` | `csrc/*`: 使用 `longlong2` 替代 `ulong2` 实现 128-bit `st.async` PTX 传输，修复 MSVC LLP64 下 `unsigned long` 仅为 32-bit 导致的 64-bit 截断 |
| `src-flashmla-deepseek-v41-kernels-2026` | `deepseek-ai/FlashMLA` `#221` (`07a1089857`) | 2026-09-10 | `implementation` | `csrc/sm90/*`: 新增面向 DeepSeek-V4.1 的 MLA 高吞吐内核 |
| `src-flashmla-decode-combine-num-splits-256-2026` | `deepseek-ai/FlashMLA` `#199` (`15f13e5030`) | 2026-07-28 | `implementation` | `csrc/*`: 将 Decode-Combine 的 `num_splits` 分桶上限从 128 扩展至 256（**`2.0x`**） |

## 2. 确定性实验结果（`experiments/flashinfer_route_pref_fa_bwd_zero_flashmla_v41_phase164`）

- **MoE Finalize+AllReduce 路由索引预取**：控制字等待周期从 **128** 降至 **32**（**`-75.0%`**），融合内核延迟从 **`19.60 us`** 降至 **`16.46 us`**（**`-16.0%`**）。
- **CuTe Bwd 空 Q/K 零梯度与 FlashMLA 256-Splits**：空 Q/K 未初始化 NaN 梯度元素从 **256** 清零为 **0**，MSVC LLP64 下 `st.async` 传输宽度恢复至完整 **128-bit**（`64 -> 128 bits`），Decode-Combine 最大分块数扩展 **`2.0x`**（`128 -> 256`）。
