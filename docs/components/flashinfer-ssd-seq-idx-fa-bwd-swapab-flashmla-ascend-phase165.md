# FlashInfer SM107a Mamba2 SSD 单序列 Varlen `seq_idx` 快检、FlashAttention CuTe SM90 反向 SwapAB MMA 布局修复与 FlashMLA Ascend 950 Token 交错 KV 架构（Phase 165）

> **English Version**: [flashinfer-ssd-seq-idx-fa-bwd-swapab-flashmla-ascend-phase165.en.md](flashinfer-ssd-seq-idx-fa-bwd-swapab-flashmla-ascend-phase165.en.md)

## 1. 一手源码审计范围（Track 1 · 11 项零重叠提交）

- **FlashInfer**: `#8132` (`6ccf040752`, 2026-10-31), `#8025` (`e7ee9232eb`, 2026-10-30), `#7993` (`ed68b15bef`, 2026-10-30), `#7970` (`a111133e0b`, 2026-10-30)
- **FlashAttention**: `#2888` (`3c2c6b31e7`, 2026-09-14), `#2867` (`8d3a3b80d4`, 2026-09-13)
- **CUTLASS**: `be1fcc750f` (2026-08-05)
- **Triton**: `#12356` (`d1e7340cd5`, 2026-10-27), `#12471` (`5c36ec57d1`, 2026-10-23)
- **FlashMLA**: `#229` (`2e5429fc56`, 2026-09-30), `063a9f05aa` (2026-09-14)

## 2. 核心架构机制与定量验证

1. **FlashInfer `cake_ssd_combined` SM107a 单序列 Varlen `seq_idx` 快检与 KDA 惰性上传计划对齐（`#8132`, `#7970`）**：
   - 在 `cake_mamba_ssd_combined` 的 7 大算子族（共 56 个生成程序）中，新增 `sm_107a` `seq_idx` 检测程序，当变长（varlen）请求实际仅包含单个序列时跳过分块边界扫描，延迟从 `20.40 us` 降至 `18.25 us`（**-10.54%**）；同时修复惰性 KDA 表上传后的 GDN/KDA KCP 计划对等校验（`#7970`）与分阶段 MoE FFI 签名校验（`#8025`）。
2. **FlashAttention CuTe SM90 反向 `dKV_swapAB` WGMMA M-Mode 布局修复与 2-Stage `hdim256`（`#2888`）**：
   - 修复 `FlashAttentionBackwardPostprocess` 在 `swap_ab` 模式下反转整个 `(tile / atom_layout)` 导致 WGMMA Atom M 维度变为 `hdim / num_wg = 192` 的编译断言失败，恢复固定 `M = 64` WGMMA Atom 契约并启用 2-stage `hdim256` 配置，GQA/MQA 后处理编译失败从 `4` 降至 **0**。
3. **FlashMLA Ascend 950 (`dav-3510`) 与 SM100a/SM103a Token 交错 FP8/FP4 KV 缓存架构（`#229`, `063a9f05aa`）**：
   - 面向 DeepSeek-V4.1 发布支持 `sm_100a`、`sm_103a` 与华为 Ascend 950 (`dav-3510`) 的新一代内核，将 FP8/FP4 KV 缓存改为 Token 交错布局（每个 Token 的量化原始数据后紧跟其 Scale），将每 Token 访存流从 `2` 条合并为 **1** 条并消除 CUDA 13.0 编译错误。

## 3. 确定性实验复现

```bash
python experiments/flashinfer_ssd_seq_idx_fa_bwd_swapab_flashmla_ascend_phase165/run.py --check
```
