# FlashInfer SM100 MegaMoE Flag-in-Data (`<=32` Rows), SM120 QSA NVFP4 Decode, PCIe CE Ring All-Reduce, Kimi-K3 LatentMoE Evict-First & DeepGEMM Mega-mHC (`h=4096`)

> **English Version**: [flashinfer-mega-moe-ll32-qsa-nvfp4-sm120-pcie-ce-ring-kimi-k3-mhc-h4096.en.md](flashinfer-mega-moe-ll32-qsa-nvfp4-sm120-pcie-ce-ring-kimi-k3-mhc-h4096.en.md)  
> **Evidence ID**: `ev-flashinfer-mega-moe-ll32-qsa-nvfp4-sm120-pcie-ce-ring-kimi-k3-mhc-h4096-2026`  
> **Experiment**: [`experiments/flashinfer_mega_moe_ll32_qsa_nvfp4_sm120_pcie_ce_ring_kimi_k3_mhc_h4096/run.py`](../../experiments/flashinfer_mega_moe_ll32_qsa_nvfp4_sm120_pcie_ce_ring_kimi_k3_mhc_h4096/run.py)

---

## 1. 一手源码审计范围与核心问题

本文基于 `flashinfer-ai/flashinfer` 2026-10-25 最新落地的 5 项核心提交（`#7348`, `#7313`, `#7301`, `#7312`, `#7321`），系统剖析 Blackwell (`sm_100a`/`sm_103a`) 与 Workstation/PCIe Blackwell (`sm_120a`) 上稀疏 MoE 通信/算子、NVFP4 稀疏注意力解码、拓扑感知 PCIe Copy-Engine 集合通信及超连接（mHC）融合算子的关键实现路径：

1. **SM100 BF16 Rank-Major EP8 MegaMoE Flag-in-Data (`<=32` Rows) 与 Pushed Route Codes (`#7348`, `d4d4bd06d1`)**：消除小 Batch Expert-Parallel (EP=8) 下的显式全局 Barrier 与独立路由 Kernel。
2. **SM120 Cake QSA NVFP4 Sparse Paged GQA Decode (`#7313`, `09a77503f0`)**：在无 `tcgen05`/TMEM 的 `sm_120a` (RTX PRO 6000 Blackwell, 188 SMs) 上实现 `mma.sync` 协同与 Split-KV 稀疏分页注意力解码。
3. **SM120 Pipelined Topology-Weighted PCIe CE Ring All-Reduce (`#7301`, `68e3e09ad3`)**：通过 Copy-Engine 流水线环（`COPY_ENGINE_RING_PIPED`）与 Stream Memory Operations 跨双路 CPU Socket 加速 `>=4 MiB/rank` 的 PCIe Gen5 All-Reduce。
4. **SM100/SM103 Kimi-K3 Stable LatentMoE Evict-First Decode 与 Resident TP1 Tail GEMM (`#7312`, `1c0d2be434`)**：小 Batch (`T<=128`) 下对权重使用 L2 `evict_first` 保护激活驻留；大 Batch (`T>=1024`) 下使用设备计数器工作认领（claimed-item）的 `256x512` 融合 RMSNorm 持久尾投影。
5. **SM100/SM103 DeepGEMM Mega-mHC (`h=4096`) 单次持久启动 (`#7321`, `847172573a`)**：将 GLM-5.3-Flash 的 4-Route Manifold-Constrained Hyper-Connection（Post-Mixing + BF16x3 Split-K 映射 + 20 步 Sinkhorn 迭代 + Collapse + Input RMSNorm）融合进单次 Persistent Kernel。

---

## 2. 关键实现路径与数据移动分析

### 2.1 SM100 MegaMoE BF16 Rank-Major EP8 Flag-in-Data (`_LL_MAX_T = 32`) (`#7348`)

在 `flashinfer/moe_ep/kernel_src/blackwell_bf16_rank_major/session.py` 与 `src/flashinfer_blackwell_moe_ep_layer_sm100.cu` 中（形状 `H=7168, I=2048, E=256` 即每卡 32 专家，`top_k=8`, `EP=8`, `1 <= T <= 128`）：
- **Flag-in-Data 低延迟路径 (`T <= _LL_MAX_T = 32`)**：Dispatch 与 Combine 阶段不再经过跨卡全局 Barrier。发送方每次直接向目标 Rank 的接收 Bank 写入 8 字节原子字（`4B payload + 4B epoch`），接收方直接轮询该 8 字节字。两个接收 Bank (`_LL_BANKS = 2`) 按 Epoch 奇偶交替，通过 System-Scope Release/Acquire 保证重用顺序，并在 32-bit Epoch 回绕前清空对侧 Bank。整层前向仅需 **6 次 Kernel 启动**，且在 `bind_weights` 完成后，`stage_inputs` 即刻将输入行异步推送至目标 Rank。
- **Pushed Route Codes (`32 < T <= 128`)**：当 `T > 32` 时（`_ROUTE_IN_DISPATCH_MAX_T = 128`），`input-arrive` Kernel 直接向各目标 Rank 推送 `route_codes`，Dispatch Kernel 使用额外 1 个 CTA 在片上直接推导路由映射，从而裁撤了独立的 `route` Kernel（前向启动数降至 **8 个 Kernel**）。此外，`unpermute` Kernel 在 Grid-Dependency Wait 之前将 `top-k` 权重与逆置换索引预取至 L2，并在 Wait 之后才读取。
- **作者实测性能（8x B200 `sm_100a`, CUDA 13.4）**：`all-ranks` 路由下 `T=1` Eager 延迟从 `164.6 us` 降至 `143.4 us` (**1.15x**)，CUDA Graph 延迟从 `155.5 us` 降至 `141.2 us` (**1.10x**)；`random` 路由下 `T=1` Eager 延迟从 `194.1 us` 降至 `174.2 us` (**1.11x**)，输出在全部 8 个 Rank 上保持 **Bit-Identical**。

### 2.2 SM120 Cake QSA NVFP4 稀疏分页 GQA 解码 (`#7313`)

在 `flashinfer/experimental/cake_qsa_nvfp4/` 中，针对 Qwen Gated Sparse Paged GQA Decode（`Hq=24, Hkv=2, D=256`，每行从 NVFP4 KV Cache 中选取 `2051` 个 Token，按 `64-token` Tile 共 33 个 Tile 组织，Page Size `16`）：
- 由于 `sm_120a`（RTX PRO 6000 Blackwell, 188 SMs）不包含 `tcgen05` 指令与 Tensor Memory (TMEM)，该提交专门交付了基于 `mma.sync` 的 K-first Decode、Cooperative Fused Decode（在单次 Launch 内通过协同归约完成 Split-KV Merge）以及独立的 Split-KV Merge Kernel。
- **作者实测性能（RTX PRO 6000 Blackwell `sm_120a`）**：相比 FlashInfer 默认 `qsa_nvfp4` 参考实现，Eager 模式下在 `rows=1..128`（NHD / HND 布局）取得 **1.27x – 1.89x** 加速（HND `T=128` 从 `320.93 us` 降至 `169.62 us`，**1.89x**）；CUDA Graph 模式下取得 **1.21x – 1.90x** 加速（HND `T=128` 从 `321.59 us` 降至 `169.10 us`，**1.90x**）。

### 2.3 SM120 拓扑加权 PCIe Copy-Engine 流水线环 All-Reduce (`#7301`)

在 `csrc/pcie_ipc_all_reduce.cu` 与 `flashinfer/comm/pcie_ipc_tuning.py` 中，新增第 8 种 PCIe IPC All-Reduce 调度方案 `IpcVariant.COPY_ENGINE_RING_PIPED`（面向 4 卡或 8 卡 `sm_120a` 且每 Rank 分片 `>= 4 MiB`）：
- 将每 Rank 的 All-Reduce 块切分为 2–4 个流水线 Piece，利用 DMA Copy Engine 配合 Stream Memory Operations 发布块到达标志，并通过 `set_ce_slow_hops` 感知跨双路 CPU Socket 的慢速 PCIe Root Complex 跳数。
- **作者实测性能（8x SM120 PCIe Gen5 跨双 Socket）**：在 TP=8 BF16 `96 MiB/rank` 下，3-Piece `RING_PIPED` 将单次 All-Reduce 延迟从 `RING_MEMOP` 的 `4.4753 ms` 降至 `4.0287 ms` (**1.111x**)，FP16 `96 MiB/rank` 从 `4.5031 ms` 降至 `4.0560 ms` (**1.110x**)。

### 2.4 Kimi-K3 Stable LatentMoE Evict-First 与 Resident TP1 Tail GEMM (`#7312`)

在 `flashinfer/experimental/kimi_k3_latent_moe/cake_backend.py` 中：
- **小 Batch (`T <= 128`) `a_evict_first` L2 驻留策略**：Front 投影（Router Logits + Latent Down + Shared-Expert Gate/Up）与 TP1 Tail 投影（RMSNorm + Latent Up + Shared-Expert Down）在 `tcgen05` Weight-Streaming 时对权重设置 `evict-first` L2 策略，防止流式权重冲刷掉容量较小的激活张量。在 `T=64` 下，Front TP1 延迟在 B300/GB300 上从 `54.02/53.07 us` 降至 `44.98/44.18 us` (**1.201x**)，Tail TP1 在 GB300 上从 `34.98 us` 降至 `29.36 us` (**1.191x**)。
- **大 Batch (`T >= 1024`) Claimed-Item Resident TP1 Tail GEMM**：CTA Pair 通过设备原子计数器动态认领 `256x512` Work Items，将 RMSNorm 与持久 2-CTA GEMM 融合，在 `T=1024` GB200 上将延迟从 `120.87 us` 降至 `107.81 us` (**1.121x**)。

### 2.5 SM100/SM103 DeepGEMM Mega-mHC (`h=4096`) 融合算子 (`#7321`)

在 `flashinfer/experimental/deepgemm_mega_mhc/mega_mhc.py` 中，针对 GLM-5.3-Flash 的 `hidden_size=4096` 四路流形约束超连接（4-route mHC），单个持久 Kernel 依次在片上完成：Post-Mixing $\rightarrow$ `BF16x3` Split-K 映射投影 $\rightarrow$ **20 轮 Sinkhorn 双随机矩阵归一化迭代** $\rightarrow$ Route Collapse $\rightarrow$ Input RMSNorm，并提供 `bf16`、`tailpipe`、`tailpipe_wsmem`、`fp8_group128` 与 `nvfp4` 五种输出变体。
- **作者实测性能**：相比 vLLM `MHCFusedPostPreOp.forward_cuda` 多 Kernel 序列，在 `T=1024` 下 B200 (`sm_100a`) 延迟从 `39.095 us` 降至 `27.424 us` (**1.426x**)，GB300 (`sm_103a`) 延迟从 `35.510 us` 降至 `25.608 us` (**1.387x**)。

---

## 3. 核心指标汇总表

| 子系统 / 算子路径 | 源码提交 | 硬件平台 | 基准延迟 ($\mu\text{s}$ / $\text{ms}$) | 优化后延迟 ($\mu\text{s}$ / $\text{ms}$) | 加速比 |
|---|---|---|---|---|---|
| MegaMoE BF16 EP8 Flag-in-Data (`T=1`, Eager) | `#7348` (`d4d4bd06d1`) | 8x B200 (`sm_100a`) | $164.6\ \mu\text{s}$ | $143.4\ \mu\text{s}$ | **1.15x** |
| Cake QSA NVFP4 Decode (`T=128`, HND, Graph) | `#7313` (`09a77503f0`) | RTX PRO 6000 (`sm_120a`) | $321.59\ \mu\text{s}$ | $169.10\ \mu\text{s}$ | **1.90x** |
| PCIe CE Ring Piped (`TP=8`, BF16, `96 MiB/rank`) | `#7301` (`68e3e09ad3`) | 8x SM120 PCIe Gen5 | $4.4753\ \text{ms}$ | $4.0287\ \text{ms}$ | **1.111x** |
| Kimi-K3 LatentMoE Front (`T=64`, `evict_first`) | `#7312` (`1c0d2be434`) | B300 / GB300 (`sm_103a`) | $53.07\ \mu\text{s}$ | $44.18\ \mu\text{s}$ | **1.201x** |
| DeepGEMM Mega-mHC (`h=4096`, `T=1024`) | `#7321` (`847172573a`) | B200 (`sm_100a`) | $39.095\ \mu\text{s}$ | $27.424\ \mu\text{s}$ | **1.426x** |

---

## 4. 架构师计算练习与可证伪问题

1. **Flag-in-Data Bank 容量与 Epoch 回绕安全边界计算**：在 `#7348` 中，`_LL_CAPACITY_T = 32`，每个 Token 发送 8 字节字（`4B payload + 4B epoch`），使用 2 个接收 Bank (`_LL_BANKS = 2`)。若单步 Decode 周期为 $141.2\ \mu\text{s}$，计算 32-bit 无符号 Epoch 计数器发生回绕所需的最短连续运行时间（$2^{31} \times 141.2\ \mu\text{s} \approx 84.2\ \text{hours}$），并解释为何每步清空对侧 Bank 可彻底消除陈旧 Epoch 误判。
2. **后续可证伪问题**：在 `sm_120a` 双路 PCIe Gen5 拓扑中，当每 Rank All-Reduce 分片小于 $4\ \text{MiB}$（如 $1\ \text{MiB}$）时，`COPY_ENGINE_RING_PIPED` 的多段切分与 Stream MemOp 门铃开销是否会劣于单段 `RING_MEMOP`？
