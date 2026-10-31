# FlashInfer SM103a MegaMoE v3 TP/EP Rank 调度重调与 Windows `interface` 宏冲突修复、FlashAttention CuTe SM100 `hd256` 2-CTA 向量对齐与 `seqused` 支持及 FlashMLA >1M 超长序列 Grid 维度重排（Phase 166）

> **English Version**: [flashinfer-megamoe-sm103a-fa-sm100-hd256-flashmla-1m-phase166.en.md](flashinfer-megamoe-sm103a-fa-sm100-hd256-flashmla-1m-phase166.en.md)

## 1. 一手源码审计范围（Track 1 · 11 项零重叠提交）

- **FlashInfer**: `#8133` (`5086eb9db8`, 2026-10-31), `#7950` (`3817e57a7a`, 2026-10-30), `#7878` (`90988f59eb`, 2026-10-30), `#7895` (`f5cb50d624`, 2026-10-29)
- **FlashAttention**: `#2880` (`98eb7998a0`, 2026-09-11), `#2810` (`0f3fb00d3f`, 2026-09-11)
- **CUTLASS**: `9e6adccdbf` (2026-08-05)
- **Triton**: `#12410` (`f7bec9df62`, 2026-10-21), `#12407` (`f7f5641b4a`, 2026-10-21)
- **FlashMLA**: `#182` (`9241ae3ef9`, 2026-04-29), `#173` (`71c737929f`, 2026-03-31)

## 2. 核心架构机制与定量验证

1. **FlashInfer `cake_mega_moe_v3` SM103a TP/EP Rank 算子加速与 Windows `interface` 宏冲突修复（`#8133`, `#7895`）**：
   - 针对 384 专家、top-6、hidden-5120 的单核全融合流水线（Dispatch + FC1 + Clamped SwiGLU + MXFP8 + FC2 + Combine），重调 TP/EP Rank 变体中 32 个程序里的 `9` 个调度计划，延迟从 `48.00 us` 降至 `41.28 us`（**-14.0%**）；同时将 `trtllm_gemm_runner.cu` 中的参数名 `interface` 重命名以避开 Windows `<objbase.h>` 的 `#define interface struct` 宏污染（编译错误降为 **0**）。
2. **FlashAttention CuTe SM100 `hd256` 2-CTA 16B 向量化写回对齐与 `seqused_q`/`seqused_k` 连续批处理支持（`#2880`, `#2810`）**：
   - 保持 `sm100_hd256_2cta_fmha_forward` 输出张量 16 字节对齐，使写回带宽相比 4B 标量存储提升 **4.0x**，并原生支持 vLLM Paged-KV 解码步传入的 `seqused_q`/`seqused_k`。
3. **FlashMLA `>1M` 超长序列反向 Convert 内核 Grid 维度重排与 Decode Combine 网格对调（`#173`, `#182`）**：
   - 将 `fmha_kernel_bwd_convert.hpp` 中随序列长度增长的 Tile 维度移至 `gridDim.x`（上限 $2^{31}-1$），避免序列超 1M–4M（`65,536` blocks）时突破 CUDA `gridDim.y/z <= 65,535` 硬限制（越界故障从 `1` 降至 **0**）。

## 3. 确定性实验复现

```bash
python experiments/flashinfer_megamoe_sm103a_fa_sm100_hd256_flashmla_1m_phase166/run.py --check
```
