# Phase 111 Track A：FlashInfer Frost W8A16 MoE 与 Hopper MSA Top-K、FA4 SM100 稀疏 MLA 原位重算 P 与分块反向，以及 TileLang `call_extern` 裸指针写标记

> **公开来源与证据边界**：本文基于公开 GitHub 仓库一手提交记录与合成离线标准库实验（`experiments/flashinfer_frost_w8a16_fa4_sparse_mla_tilelang_phase111/run.py`），严格区分**规范（Specification）**、**代码实现（Implementation）**、**上游作者测量（Author Benchmark）**与**本项目离线模型验证（Own-Offline）**，不宣称已在真实 SM90/SM100/SM103/SM120 硬件上重跑 GPU Profiling。

---

## 1. FlashInfer 算子演进：Frost W8A16 MoE、Hopper MSA Top-K、动态 EP 通信上限、SM120 NVFP4 投影、GB300 DSA 阈值、SM12x >2GiB 静态 DLPack 与非 64 倍数 SSU

### 1.1 Frost 低延迟分组 W8A16 MoE 候选集（`flashinfer-ai/flashinfer@d42c276bd872`, 2026-10-27）
- **实现路径**：`flashinfer/jit/gemm/moe/gen_moe_w8a16_module.py` 在 `E4M3` 权重 + `BF16/FP16` 激活的分组 Tensor Core MoE 内核中扩展了跨三代架构的候选：
  - **SM90 (`sm_90a`)**：在 `T4/T8` BF16 GeGLU 形状上引入 `K256` tile；
  - **SM103 (`sm_103a`, GB300)**：为 `T4` BF16 GeGLU 启用 `serial_gather=True`，并补充 `T8/H7168/I2048/K8` DeepSeek-R1/V3 拓扑；
  - **SM120 (`sm_120a`, RTX 5090 / RTX PRO 6000)**：引入 `group8 / stages2 / K128` 流水线配置。

### 1.2 Hopper `cake_hopper_msa` 单遍精确 Top-K 块选择（`flashinfer-ai/flashinfer@3d01e3fec63f`, 2026-10-27）
- **实现路径**：在 `csrc/cake_hopper_msa_jit_binding.cu` 与 `flashinfer/data/csrc/cake_hopper_msa/` 中将 3 个 `msa_topk_select` 候选重新生成到单遍精确调度（Single-Pass Exact Schedule）：
  - `276e272a54b4efd20df6`：无掩码 1-chunk、128-wide、11-level；
  - `7bbffa90057de145b905`：带掩码 1-chunk、128-wide、11-level；
  - `e9b601567cc3a6dd8033`：带掩码 2-chunk、32-wide、9-level 低延迟变体（作者在 H100 上报告最高 `1.06x` 加速）。

### 1.3 `MoEEpCommunication.dispatch` 动态步长 Token 上界（`flashinfer-ai/flashinfer@47e64b0e96c4`, 2026-10-27）
- **实现路径**：`flashinfer/comm/moe_ep.py` 为 `MoEEpCommunication.dispatch()` 与 `MoEEpSplitLayer` 增加可选参数 `runtime_max_tokens_per_rank: Optional[int] = None`，并校验：
  $$\text{num\_tokens} \le \text{runtime\_max\_tokens\_per\_rank} \le \text{max\_tokens\_per\_rank}$$
- **架构影响**：当解码批次远小于预分配缓冲池上限 `max_tokens_per_rank`（例如 `512` vs `4096`）时，动态收紧 per-step 分发上界可将 All-to-All / IPC 步进有效扫描跨度压缩 `8.0x`，同时无需重新分配静态对称内存。

### 1.4 SM120 MiniMax-H3 `quant_out_proj` NVFP4 路由与 GB300 DSA Indexer 采样阈值偏移
- **SM120 NVFP4 170-SM 路由（`flashinfer-ai/flashinfer@73c83aac3125`, 2026-10-27）**：`flashinfer/gemm/kernels/cake_minimax_h3_sm120_quant_out_proj.py` 将 `PROGRAM_1_SMS` 扩展为 `{0: (188,), 1: (188, 170)}`。在 170-SM 的 GeForce RTX 5090 上运行 NVFP4（`quant=1`）时，路由至 `Program 1`（`warp_epilogue`），每对环形流水阶段（pair of ring stages）仅读取一次激活缩放行（activation-scale row），作者报告在 `M=4097`（`346.3 -> 338.8 us`, `1.022x`）与 `M=33472`（`2316.9 -> 2267.1 us`, `1.022x`）均获得稳定收益。
- **GB300 `cake_dsa_indexer` 采样偏移规则（`flashinfer-ai/flashinfer@2b5a5ffff99a`, 2026-10-27）**：在 `sm_103a`（GB300）上注入 `sample_shift_rule = [4096, 375]`，当 `top_k >= 4096` 时将首轮采样阈值偏移量从 `250‰` 提升至 `375‰`，提前过滤更多被淘汰块，作者在 GB300 `Z1k4096` 上实测 `1.13x` 加速。

### 1.5 SM12x W4A16 MoE `> 2 GiB` 静态 CuTe DLPack 布局与非 64 倍数 `head_dim` SSU
- **SM12x W4A16 `> 2 GiB` 权重池防溢出（`flashinfer-ai/flashinfer@9c4aacf1bfd6`, 2026-10-26）**：在 Qwen3.6-35B-A3B-NVFP4（`256` 专家，`inter_x2=2048`，`packed_k=5120`）中，打包后的 `w13` 扁平 `uint8` 字节数达到 $256 \times 2048 \times 5120 = 2,684,354,560\text{ B} > 2^{31}-1$。`_w4a16_fused_moe_launch_flat` 改用 `cute.runtime.from_dlpack(w13_arg, assumed_align=16)` 以静态布局传入，规避 `mark_layout_dynamic()` 将步幅压入 32 位有符号整数导致的 `OverflowError`。
- **非 64 倍数 `head_dim` Mamba SSU（`flashinfer-ai/flashinfer@0c3d03c7656a`, 2026-10-27）**：`dispatchCtasPerHead` 将编译期可分性条件收紧为 `kDim % CTAS == 0 && (kDim / CTAS) % kMinRows == 0`（其中 `kMinRows = 16`），使 `head_dim=96` 自动选 `2` 个 CTA/head（`48` 行/CTA）、`head_dim=80` 自动选 `1` 个 CTA/head（`80` 行/CTA），并在 epilogue 处增加 `elemsPerThreadEpilogue % load_output_t::count == 0` 守卫（覆盖 `head_dim=288` BF16）。

---

## 2. FlashAttention-4 SM100 稀疏 MLA 原位重算 P 与分块反向（`Dao-AILab/flash-attention@294cfb3593ed`, `#2816`）及 `HD256` 变长调度（`38ddafeb35e5`, `#2807`）

### 2.1 稀疏 MLA 训练显存解剖与双正交优化
在 DSA 风格稀疏 top-k MLA（`q_heads=128`, `topk=2048`）训练步中：
- 默认前向每 token 需保存未归一化概率 `p`（BF16 `[T, 128, 2048]` = $512\text{ KiB/token}$）与 `row_max`（FP32 = $8\text{ KiB/token}$），在 $T = 65,536$（`64K`）时每层仅保存的激活 `p` 即高达 **32.0 GiB**；反向传播还会额外物化同尺寸的瞬态张量 `dS`（**32.0 GiB**），导致单步注意力峰值达 `~82 GiB`。
- **`gather_bwd_recompute_p=True`**：前向仅保存 `out + lse`（`p` 与 `row_max` 降为 **0 GiB**，因消除了 softmax 关键路径上的 `p` 写回，`8K` 前向耗时从 `4.43 ms` 降至 `3.77 ms`）；反向主内核通过一次额外的 `S^T` UMMA（在已 gather 的 `V` 与 `K_rope` 上计算）原位重建：
  $$P^T = \exp_2\left(\text{scale}\cdot\log_2 e \cdot S^T - \text{lse}\cdot\log_2 e\right)$$
  并在启动时将 `sV` gather 阶段清零，防止被掩码的 topk 槽位跳过 `cp.async` 后与共享内存垃圾值产生 `0 * NaN`。
- **`gather_bwd_token_chunk=N`**：将反向三内核按 token 分块执行（例如 `N = 4096` 时 `dS` 瞬态从 `32.0 GiB` 降至 **2.0 GiB**）。在因果掩码下，每个分块 `[tok0, tok1)` 将其有效 K 边界收缩为：
  $$k_{\text{end}} = \text{seqlen\_k} - \text{seqlen\_q} + \text{tok}_1$$
  从而精确复现前向因果位掩码，且当 $k_{\text{end}} \le 0$ 时直接跳过全部三个反向内核。

### 2.2 SM100 `HD256` 变长网格上界与对齐向量化 Epilogue（`38ddafeb35e5`）
- 修复了 `head_dim == 256` 专用 SM100 2-CTA FMHA 内核在变长（varlen）输入下按 `batch_size * ceil(total_q / tile_m)` 过度启动 CTA 的问题：无主机侧 `max_seqlen` 提示时启用 `SingleTileVarlenScheduler` 扁平调度，有提示时按 `max_seqlen_q` 约束网格，并用页表行宽 `page_table.shape[1] * page_size` 约束 `max_seqlen_k`。

---

## 3. TileLang `AnnotateReadOnlyParams` 针对 `call_extern` 裸数据变量的写标记修复（`tile-ai/tilelang@f3ffeaa6b8ee`, `#2618`）

- **问题根因**：`src/transform/annotate_read_only_params.cc` 中的 `ReadWriteMarker` 此前仅识别 `address_of(Buf[...])` 形式的指针写入。当用户向外部函数直接传递裸缓冲区指针（如 `T.call_extern("void", "store_val", Out.data, val)`）时，编译器误将 `Out` 标记为只读并生成 `const float* __restrict__ Out`，引发 CUDA 编译失败或未定义行为。
- **修复机制**：当 `op->op.same_as(builtin::call_extern())` 为真且实参为 `param_or_data_vars_` 中的裸 `VarNode` 时，保守地将其加入 `written_` 集合；而具有已知纯函数语义的内建算子（如 `tirx.isnullptr`）仍保留只读优化。

---

## 4. 本项目可复现实验验证（`experiments/flashinfer_frost_w8a16_fa4_sparse_mla_tilelang_phase111`）

运行 `python experiments/flashinfer_frost_w8a16_fa4_sparse_mla_tilelang_phase111/run.py --check` 输出关键指标：
- `MoEEpCommunication` 动态 `runtime_max_tokens_per_rank=512`（对比静态 `4096`）将单步分发扫描体积降低 **8.0x**；
- SM12x W4A16 256 专家权重池（`2,684,354,560 B`）超出 32 位有符号整数上限（`2,147,483,647`），静态 `from_dlpack` 消除溢出；
- Mamba SSU `dispatchCtasPerHead` 对 `head_dim = {64: 4, 80: 1, 96: 2, 128: 4, 288: 4}` 精确匹配每趟 16 行整除约束；
- FA4 SM100 稀疏 MLA 在 `64K` 序列下启用 `gather_bwd_recompute_p=True` 与 `gather_bwd_token_chunk=4096` 后，保存概率激活从 **32.0 GiB 降至 0 GiB**，反向 `dS` 瞬态从 **32.0 GiB 降至 2.0 GiB**。
