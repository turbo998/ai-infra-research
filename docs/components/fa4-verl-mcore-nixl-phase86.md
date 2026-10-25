# FlashAttention-4 1CTA MLA 与三级 dV 流水、veRL 截断支撑集得分中心化、Megatron 动态行 MoE、Mooncake EFA THP 与 XPU dma-buf、NIXL GPUNetIO 双 SGE、Dynamo 分层前缀及 TE 融合 MLA KV RoPE MXFP8 深度解析（Phase 86）

> **Language / 语言**: **中文 (当前)** | [English](fa4-verl-mcore-nixl-phase86.en.md)

## 1. 背景与跨层协同演进

在 Blackwell (`SM100`/`SM103`/VR200) 注意力与量化算子、强化学习后训练（RLHF/RLVR）、确定性 MoE 推理以及 RDMA/GPUNetIO 分布式 KV 传输栈中，Phase 86 Track 3 审计了 7 个核心上游仓库的 10 项里程碑提交：

1. **FlashAttention-4 (`Dao-AILab/flash-attention`) SM100 1CTA MLA 前向/反向体系与 `hd256` 无信号量确定性反向**（[PR #2952](https://github.com/Dao-AILab/flash-attention/commit/3451a2a67a24eeed8af54d3c5b8d219577113797)、[PR #2938](https://github.com/Dao-AILab/flash-attention/commit/2a92225a91385431ec8b3d095f8d7676d7a48e82)）。
2. **TransformerEngine (`NVIDIA/TransformerEngine`) 融合 MLA KV 拆分 + Key RoPE + 双视图 MXFP8 量化**（[PR #3672](https://github.com/NVIDIA/TransformerEngine/commit/459dedc945a36abbb728479c165acc8a2c196968)）。
3. **veRL (`volcengine/verl`) 截断采样支撑集（Truncated Sampling Support）精确得分中心化**（[Commit `08b8387`](https://github.com/volcengine/verl/commit/08b83872283edffb7f710a9bce26bf2a3d6549a7)）。
4. **Megatron-LM (`NVIDIA/Megatron-LM`) 打包 MTP 上下文并行边界批量交换与 Batch-Invariant MoE 活跃行裁剪**（[PR #8389](https://github.com/NVIDIA/Megatron-LM/commit/50f94464f6ae5fa7f68aac29b6c5da8d3ed16eff)、[PR #8385](https://github.com/NVIDIA/Megatron-LM/commit/77232c387ad51e3e0724cf9da9bf50306db98015)）。
5. **Mooncake EFA 2MB THP 页表预算与 Intel XPU Level Zero `dma-buf` GPUDirect RDMA、NIXL GPUNetIO 异构 GPU 显存与双 SGE WRITE、Dynamo 按存储层追踪会话前缀驻留**（[Mooncake PR #4564](https://github.com/kvcache-ai/Mooncake/commit/088d599e62b77a4f9a65607dafd2fd4300ed8e2c)、[Mooncake PR #4378](https://github.com/kvcache-ai/Mooncake/commit/042ab8a18b0e6ccfe87782261a5556a3dd4446e4)、[NIXL PR #2172](https://github.com/ai-dynamo/nixl/commit/14e329032bff3aa3f2431985fad54e91c7da0fed)、[Dynamo PR #16501](https://github.com/ai-dynamo/dynamo/commit/c178cbae9b1a2e175a9689a6f8d91d26428675d5)）。

---

## 2. 第一手源码级实现路径与数学公式分析

### 2.1 FlashAttention-4 SM100 1CTA MLA、三级 TMEM `dV` 流水与 TE 融合 MLA KV RoPE MXFP8

- **SM100 `head_dim=256` 专用反向天然确定性（PR #2952）**：
  - 在 `flash_attn/cute/interface.py` 与 `sm100_hd256_2cta_fmha_backward.py` 中，专用 `hd256` 反向的每个 `dQ, dK, dV` Tile 均在单 CTA 内部按固定顺序归约（无跨 CTA `atomicAdd`），因此移除 `assert not deterministic` 并跳过 `dQ_semaphore / dK_semaphore / dV_semaphore` 分配，使 `head_dim=256` 模型直接支持 `deterministic=True` 位级一致训练。
- **SM100 1CTA MLA 前向/反向体系（PR #2938）**：
  - **1CTA `kb64` 主循环与波次分发启发式（`_mla_1cta_route`）**：64-row Tile（1 token $\times \le 64$ heads）在 GB300 上实现 **`6.9 TB/s`** 有效 KV 载荷带宽（达 HBM3e 标称带宽的 ~86%，超越 2CTA 的 `6.2 TB/s`）。在未切分（`num_splits=1`）Dense Decode 与 Sparse Top-K Gather 中，当 $2 \times \text{tokens} \times H_{kv} \le 152$（GB300 单波次 SM 数，对应 $\le 76$ tokens）时保留 2CTA，超过单波次（$\ge 80$ tokens）切换至 1CTA（提速 `1.07x–1.45x`）。
  - **`ptxas -O2` 消除局部内存溢出**：默认 `ptxas` 优化级别下 `dQdQvGemmKernel` 每线程溢出 **`4064 B`** local memory；为 MLA 内核类绑定 `ptxas_options = "-O2"` 后溢出降为 **`0 B`**，在 GB300 上提速 **`3.0x–4.2x`**。
  - **64 头稀疏 MLA 反向 3 级 TMEM `dV` 累加环**：在 64 头下将 `num_stages_dV` 从 `2` 增至 `3`（占用 $3 \times 128 + 32 (\text{dP}^T) + 32 (\text{S}^T) = 448 \le 512$ 列 TMEM，剩余 $448..511$ 列供融合 `dK_rope` 使用），将 `(group g, split s)` 映射至阶段 $(2g + s) \bmod 3$，使 `mma_dV_leg1(g+1)` 无需等待 Epilogue 排空并 `red.global.add.v4` 散射第 $g$ 组，在 $T=S=16\text{K}$ 上从 `18.32 ms` 降至 **`15.29 ms`（1.198x）**。
  - **任意头数（$1..64$）稀疏 MLA `recompute-P` 反向**：用 `qheads_first_tma_view` 对填充行补零，并将 `lse_log2` 填充行初始化为 $+\infty$，使填充行精确满足 $P = \exp_2(0 - \infty) = 0, \text{dS} = 0$。24 头 $16\text{K}$ tokens 训练步耗时从 2CTA `load-P` 的 `30.2 ms` 降至 **`22.5 ms`（1.342x）**。
- **TransformerEngine `nvte_fused_mla_kv_rope_mxfp8`（PR #3672）**：
  - 在 `fused_mla_kv_rope_mxfp8.cu` 中，单 CTA 处理 32 tokens $\times$ 2 heads（`kPitch = 192 + 8 = 200` BF16 错开 Shared Memory Bank），一次性读取 `kv [s, b, h, 256]` 与共享 `k_rope [s, b, 64]`，在 SMEM 中完成 `k_nope (128) + RoPE(k_rope) (64)` 拼接与 `v (128)` 拆分，随后由 6 个 Warp 行向（32 元素块）+ 3 个 Warp 列向（32 token 块）直接量化写出紧凑 E8M0 缩放的 MXFP8 `key [s, b*h*192]` 与 `value [s, b*h*128]`。在 VR200 `[4096, 1, 128 heads]` 上从 `60.8 + 75.6 = 136.4 us` 降至 **`100.1 us`（1.363x）**。

### 2.2 veRL 截断采样支撑集得分中心化与 Megatron-LM 确定性 MoE 活跃行边界

- **veRL `score_centering_support_correction`（Commit `08b8387`）**：
  - 当 vLLM 采样启用截断（`top_p < 1` 或 $0 < \text{top\_k} \le \text{topk\_log\_probs}$）时，vLLM 的 `processed_logprobs` Top-$k$ 头部完整包含支撑集 $S = \{i : \log q_i > -\infty\}$ 且满足 $\sum_{i \in S} q_i = 1$。
  - `score_centering_support_correction` 在 $S$ 上对训练策略重归一化：
    $$\log p(S) = \operatorname{logsumexp}_{i \in S}(\log p_i), \qquad p_i^S = \frac{p_i}{p(S)} \quad (i \in S)$$
    $$\mathcal{L}_{\text{center}}^S = \sum_{i \in S} \operatorname{sg}\!\left[q_i w\!\left(\frac{p_i^S}{q_i}\right) - \left(\sum_{j \in S} q_j w\!\left(\frac{p_j^S}{q_j}\right)\right) p_i^S\right] \log p_i^S$$
    同时从被采样 token 的 $\log p_a$ 中减去 $\log p(S)$，使重要性采样比 $p_a^S / q_a$ 与中心化项严格定义在同一截断支撑集 $S$ 上。
- **Megatron-LM 打包 MTP CP 边界批量交换与活跃行 MoE（PR #8389 & PR #8385）**：
  - **PR #8389**：在 `multi_token_prediction.py` 中新增 `_exchange_cp_roll_boundaries`，通过 `torch.distributed.batch_isend_irecv` 在 `cp_group` 上一次性批量交换前后两个 CP 边界并保留打包序列局部接缝处的重复偏移。
  - **PR #8385**：在 `batch_invariant.py`、`permute.py` 与 `mxfp8_quantize.py` 中，将 `tl.range(pid, max_rows, NUM_BLOCKS)` 改为 `tl.range(pid, tl.minimum(n_used, max_rows), NUM_BLOCKS)`（其中 `n_used = offs[-1:]` 在设备端读取，保持 CUDA Graph 静态形状兼容），并新增 `mxfp8_quantize_rows(x, n_used)` 与 `_zero_permutation_padding_kernel`，在 `1280 / 16384` 活跃行场景下将量化与反置换工作量降低 **`12.8x`**。

### 2.3 Mooncake EFA 2MB THP / XPU dma-buf、NIXL GPUNetIO 双 SGE 与 Dynamo 分层前缀

- **Mooncake AWS EFA 2MB THP 页表项预算与三级回退（PR #4564）**：
  - Linux `/proc/self/smaps` 对透明大页（THP）仍报告 `KernelPageSize: 4 kB`，导致 EFA 传输层按 4KB 预算 `kDefaultMaxPteEntries = 22M` 页表项，单网卡上限仅 $22\text{M} \times 4\text{ KiB} = \mathbf{88.0\text{ GiB}}$。
  - 修复 `parseVmaHeader`（避免将 `"AnonHugePages:"` 首字母 `'A'` 误判为十六进制 VMA 头），对 `MADV_HUGEPAGE` 内存先 `preTouchMemory`（含尾部余数），验证 `Rss == AnonHugePages` 后按 **2MB** 预算 PTE（单网卡理论上限提升至 **`44.0 TiB`**），并在 `fi_mr_reg` 返回 `ENOMEM` 时按 `kAuto (2MB 全覆盖) -> kThpPartitioned (2MB 按网卡分片) -> kBasePages (4KB 分片)` 顺序优雅降级。
- **Mooncake Intel XPU Level Zero `dma-buf` GPUDirect RDMA（PR #4378）**：
  - 在 `tent/src/platform/xpu/` 中通过 `ZeLoader` 动态解析 `libze_loader.so.1` 的 `zeMemGetAllocProperties` 与 `zeMemGetAddressRange`，导出 `ZE_IPC_MEMORY_FLAG_TBD` / `dma-buf` fd 并通过 `ibv_reg_dmabuf_mr` 注册；通过 `hasDedicatedDmabuf` 拦截未关闭 USM 池化（`kExportPooled`）的歧义句柄并安全回退至 Host Staging。
- **NIXL GPUNetIO 异构 Peer VRAM 注册与双 SGE WRITE 合并（PR #2172）**：
  - 支持非通信引擎所属的其它可见 GPU 显存通过 `cudaDeviceGuard(mem.devId)` 与 `ibv_reg_mr`（PeerDirect）注册；
  - 在 `canPairWrite` 中，当连续两个 WRITE 描述符满足远端地址相邻（`next_address - address == length`）、`rkey` 相同且非最后一个描述符（保留最后一个独立描述符以确保到达标记 Arrival-Marker 顺序）时，合并为单个双 SGE `doca_gpu_dev_verbs_wqe_prepare_write` WQE。在双 H20 GPU 的 $4096 \times 4\text{ KiB}$（`16 MiB`）基准测试中，数据 WQE 从 `4096` 降至 **`2049`**，中位吞吐从 `10.397 GB/s` 提升至 **`12.697 GB/s`（+22.12%）**，Tx p99 从 `1613 us` 降至 **`1325 us`**。
- **Dynamo KV Router 按存储层追踪会话前缀驻留（PR #16501）**：
  - 在 `lib/kv-router/src/session_prefix_index.rs` 中将 `FrontierKey` 扩展为 `(WorkerWithDpRank, StorageTier)`，分离 `by_rank` 内容版本与 `lifecycle_by_rank` 生命周期版本，使路由器同时感知 GPU 显存层与 CPU/SSD 下游存储层的最深会话前缀块。

---

## 3. 可复现实验验证（`experiments/fa4_verl_mcore_nixl_phase86`）

运行 `python experiments/fa4_verl_mcore_nixl_phase86/run.py --check` 验证通过：
- **FA4 & TE**：`dQdQvGemmKernel` 溢出 `4064 -> 0 B`，3 级 `dV` TMEM 占用 `448` 列（提速 `1.198x`），TE 融合核提速 `1.363x`；
- **veRL & Megatron-LM**：截断支撑集重归一化概率和 `1.000000`、中心化残差和 `0.0`，Megatron 活跃行工作量降低 `12.8x`；
- **Mooncake & NIXL & Dynamo**：EFA 单网卡容量上限 `88.0 GiB -> 45056.0 GiB`，NIXL WQE `4096 -> 2049`、吞吐提升 `+22.12%`。

## 4. 架构师设计练习与可证伪问题

1. **GPUNetIO 双 SGE WQE 计数推导**：给定 $N=4096$ 个远端连续的 4 KiB WRITE 描述符，为什么保留最后一个描述符用于到达标记后，最终生成的 WQE 数量是 $2049$ 而不是 $2048$？
2. **可证伪假设**：在 veRL 中若启用 `top_p = 0.9, top_k = 64` 但未开启 `score_centering_top_p: true`（仍使用未截断的 Head+Tail 模型），当训练策略在截断支撑集 $S$ 之外赋予较高概率质量（$p(S) < 0.7$）时，重要性采样比 $p_a / q_a$ 是否会产生系统性低估偏置？
