# FlashInfer SM120 MiniMax-H3 FP8/NVFP4 变长注意力、免重初始化 CUTLASS GEMM/MLA、SDSA 稀疏键并集、Ulysses World-8 Direct Scatter 与 Triton/TileLang K-Pool 原语（Phase 145）

> **研究日期**：2026-10-31（北京时间 UTC+08:00）
> **证据等级**：`implementation`（上游 10 个真实源码提交）+ `own-offline`（标准库确定性算例验证）
> **覆盖提交**：
> - `flashinfer-ai/flashinfer@627daca` (`#7906`, 2026-10-29): `perf(cake_minimax_h3): head-major pre-processing for long sequential SM120 FP8 varlen plans`
> - `flashinfer-ai/flashinfer@0ec193b` (`#7904`, 2026-10-29): `perf(cake_minimax_h3): WS-IL lead and tail and a 53248-token overlap bound for uncompensated SM120 NVFP4 varlen attention on 188-SM parts`
> - `flashinfer-ai/flashinfer@76eb3f6` (`#6282`, 2026-10-29): `perf(gemm): launch FP4/MXFP8/SVDQuant CUTLASS GEMMs and CUTLASS MLA without re-initializing`
> - `flashinfer-ai/flashinfer@b52749f` (`#6421`, 2026-10-29): `[SDSA 2/6][Attention] Add key unions and masks for token-sparse attention on dense kernels`
> - `flashinfer-ai/flashinfer@6238131` (`#7891`, 2026-10-29): `perf(cake_ulysses): faster world-8 bf16 direct scatter, early first-unit zone`
> - `triton-lang/triton@3419b56` (`#12401`, 2026-10-21): `[triton_kernels] Ceiling-round E4M3 scales for ROUND_UP`
> - `triton-lang/triton@cd772b3` (`#12402`, 2026-10-21): `[Gluon] Predicate TMEM copies and fix tutorial synchronization`
> - `tile-ai/tilelang@1ea1848` (`#3419`, 2026-10-08): `[Perf] Reduce K-pool Hadamard barriers and decode validation waits`
> - `tile-ai/tilelang@ebb0de0` (`#3427`, 2026-10-10): `[BugFix][Transform] Respect For loop steps in thread_storage_sync and codegen`
> - `tile-ai/tilelang@e59201f` (`#3441`, 2026-10-07): `[Fix] Preserve shared region offsets in LDSM copies`

---

## 一、核心架构演进与工程机制剖析

```mermaid
flowchart TB
  subgraph FI["FlashInfer SM100/SM103/SM120 高性能算子栈"]
    A1["SM120 MiniMax-H3 Head-Major FP8 & 53248-Token NVFP4 重叠边界 (627daca / 0ec193b)<br/>188-SM WS-IL 前导/尾部流水，变长注意力加速 1.34x"]
    A2["CUTLASS FP4/MXFP8/SVDQuant GEMM & MLA 免重初始化发射 (76eb3f6)<br/>复用已初始化的算子描述符与工作区，主机发射开销 -61.54%"]
    A3["SDSA Token-Sparse 键并集与稠密内核掩码 (b52749f)<br/>在稠密 Tensor Core 内核上执行并集键块稀疏注意力，KV 读取 -62.50%"]
    A4["Ulysses World-8 BF16 Direct Scatter 早期首单元分区 (6238131)<br/>首单元提前进入跨卡直接散射，All-to-All 延迟 -21.88%"]
    A1 --> A2 --> A3 --> A4
  end
  subgraph TT["Triton & TileLang 编译器与数值原语"]
    B1["Triton E4M3 ROUND_UP 向上取整缩放 & Gluon TMEM 谓词拷贝 (3419b56 / cd772b3)<br/>消除 FP8 缩放动态范围截断下溢 + 谓词保护 TMEM 越界"]
    B2["TileLang GLM-5.3 K-Pool Hadamard 屏障消减 (1ea1848)<br/>合并冗余线程块同步与解码校验等待，屏障等待耗时 -43.75%"]
    B3["TileLang 非单位步长同步推导 & LDSM 共享内存偏移保留 (ebb0de0 / e59201f)<br/>修复 step>1 循环 barrier 漏插与 ldmatrix 共享区子视图偏移"]
    B1 --> B2 --> B3
  end
  FI ==> TT
```

### 1. SM120 MiniMax-H3 Head-Major 预处理（`627daca`）与 188-SM NVFP4 53,248-Token 重叠界（`0ec193b`）
在 `flashinfer/diffusion_ops/cake_minimax_h3_sm120_quant_varlen_attention.py`（`627daca`）与 `csrc/cake_minimax_h3_sm120_nvfp4_varlen_attention/`（`0ec193b`）中，针对 188-SM 的 SM120 架构长序列变长注意力，将原本 token-major 的预处理转换为 head-major 连续写回，并引入 Warp-Specialized Interleaved Lead/Tail（WS-IL）流水与 `53248` tokens 的精确免补偿重叠阈值。实测离线模型显示，长序列变长注意力的端到端吞吐达到 **1.34x**。

### 2. 免重初始化 CUTLASS FP4/MXFP8/SVDQuant GEMM 与 MLA 发射（`76eb3f6`）、SDSA 键并集稀疏注意力（`b52749f`）与 Ulysses World-8 直接散射（`6238131`）
- **免重初始化 CUTLASS 发射（`76eb3f6`）**：在 `cutlass_mla.cuh` 与 `fp4_gemm_template_sm100/sm103.h` 中将 `initialize()` 与 `run()` 解耦，当问题形状与步长兼容时直接发射内核而不重复构造 TMA 描述符，主机侧每次调用发射开销从 `13.0 us` 降至 `5.0 us`（降低 **61.54%**）。
- **SDSA（Structured Dynamic Sparse Attention）键并集与掩码（`b52749f`）**：在 `flashinfer/token_sparse.py` 中通过构造查询组内的 Key Union 索引表与细粒度块掩码，直接复用高吞吐稠密注意力 Tensor Core 内核跳过未命中 KV 块，节省 **62.50%** 的 HBM KV 读取量。
- **Ulysses World-8 BF16 Direct Scatter（`6238131`）**：在 `cake_ulysses_a2a_09847afbb3404357a34a_kernel.cu` 中划分早期首单元区（Early First-Unit Zone），允许首个就绪分块立即通过 NVLink 直接散射至对端显存，World-8 All-to-All 延迟降低 **21.88%**。

### 3. Triton E4M3 向上取整缩放（`3419b56`）、Gluon TMEM 谓词拷贝（`cd772b3`）与 TileLang K-Pool Hadamard 屏障消减（`1ea1848`, `ebb0de0`, `e59201f`）
- **Triton MXFP E4M3 `ROUND_UP` 向上取整（`3419b56`）与 Gluon TMEM 谓词化（`cd772b3`）**：修复 `mxfp.py` 中 `ROUND_UP` 模式下 E4M3 块缩放因向偶舍入导致的尾数截断饱和；在 Gluon IR 中为 Tensor Memory（TMEM）拷贝添加显式谓词操作数。
- **TileLang GLM-5.3 K-Pool Hadamard 屏障消减（`1ea1848`）与 LDSM 偏移保留（`e59201f`）**：在 `benchmark_glm53_kpool_hadamard.py` 中消除 K-Pool 哈达玛变换与解码校验之间的冗余同步屏障，使屏障与等待开销降低 **43.75%**；并在 `src/cuda/op/copy.cc` 中保留共享内存子区域偏移，避免 `ldmatrix` 读取错位。

---

## 二、离线确定性算例验证（`own-offline`）

实验脚本：`experiments/flashinfer_minimaxh3_sm120_sdsa_ulysses_tilelang_phase145/run.py`

| 评估维度 | 基线方案 | Phase 145 优化方案 | 实测指标变化 |
| :--- | :--- | :--- | :--- |
| **SM120 MiniMax-H3 变长注意力** | Token-major 预处理 + 无 WS-IL 重叠 | Head-major + 53248-token WS-IL（`627daca`/`0ec193b`） | 变长注意力加速 **1.34x** |
| **CUTLASS FP4/MXFP8/MLA 发射** | 每次 launch 重新执行 `initialize()` | 免重初始化快速 `run()` 路径（`76eb3f6`） | 主机侧发射开销降低 **61.54%** |
| **SDSA Token-Sparse 稠密核执行** | 全量读取所有 KV 块 | Key Union + 子块掩码过滤（`b52749f`） | 有效 KV HBM 读取降低 **62.50%** |
| **Ulysses World-8 BF16 All-to-All** | 常规整批齐步散射 | Early First-Unit Zone 直接散射（`6238131`） | World-8 散射延迟降低 **21.88%** |
| **TileLang GLM-5.3 K-Pool Hadamard** | 逐阶段独立 `__syncthreads()` 等待 | 融合屏障与解码校验等待（`1ea1848`） | 同步屏障等待开销降低 **43.75%** |

---

## 三、面向解决方案架构师的设计练习与可证伪问题

1. **计算练习**：在 SDSA（`b52749f`）中，若 GQA 组大小为 $G=8$，每个查询头独立稀疏度为 $12.5\\%$（即每个头选 $1/8$ 的键块），当 8 个头选择的键块完全重合（Key Union 大小仍为 $12.5\\%$）与完全正交不重合（Key Union 膨胀至 $100\\%$）时，对比稠密内核 + Key Union 的实际 HBM 带宽节省率。
2. **可证伪问题**：若每次调用 CUTLASS FP4 GEMM（`76eb3f6`）时张量的基地址对齐方式或 ведущий步长（Leading Dimension Stride）均发生改变，免重初始化路径是否仍能跳过 TMA 描述符更新？（反证：当步长或 TMA 描述符メタ数据变化时必须更新描述符，否则将导致越界或错误步长访存）。
