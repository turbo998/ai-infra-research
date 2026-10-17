# AI Infra 创新公司深挖：推理加速平台（Fireworks / Together）与新型 AI 芯片（Groq / Cerebras）

状态：2026-10-18（持续更新自 2026-10-14），基于固定开源代码提交与同行评审/预印本论文正文的**有界公开研究**。
先修：[GPU/HBM 深入研究](gpu-hbm.md)、[互连与网络分层](interconnect-network.md)。
**严格边界声明**：本文仅依据公开源码（`implementation`）、架构论文（`specification` / `author-benchmark`）及本仓库合成公式推导立论；**未调用任何商业付费 API，未在 B200/H100/Groq LPU/Cerebras WSE 硬件上实测**，作者披露的跑分均显式标注为 `author-benchmark`，绝不冒充本项目独立重测（`own-hardware`）。

证据 ID：`fireworks-m3-sparse-sm100`、`fireworks-b200-author-bench`、`fireworks-locust-cache-bench`、
`fireworks-m3-ancillary-cute-pipeline`、`fireworks-prefill-load-test-routing`、
`together-fa3-hopper-async`、`together-sequoia-spec-tree`、`together-sequoia-dp-spectree-impl`、`together-sequoia-offload-engine-pcie`、`together-stripedhyena-state`、`together-moa-fanout`、
`groq-tsp-microarch-bert`、`groq-groqflow-multichip`、`groq-groqflow-multichip-runtime`、`groq-groqflow-estimate-perf`、`cerebras-csl-wse-allreduce`、`cerebras-wse-io-roofline`、
`cerebras-csl-gemv-2d-collectives`、`cerebras-csl-sync-wse3`、`cerebras-sdk-bw-fifo-compensated`、`capacity-economics-offline-sim`。
来源行范围、commit SHA 与发布日期见 [sources](../../research/sources.json)；主张与证据类型见 [evidence](../../research/evidence.json)。

---

## 1. 为什么解决方案架构师必须用“存储层次与通信原语”解构 AI Infra 创新公司？

云架构师在评估推理加速平台（如 **Fireworks AI**、**Together AI**）或非 GPU 架构芯片（如 **Groq TSP/LPU**、**Cerebras WSE**）时，最容易陷入只看“单序列峰值 Token/s”或营销宣传的误区。回到第一性原理，所有 AI Infra 创新都在求解以下三个受物理定律支配的方程：

1. **单步 Decode 的字节搬运下界（Memory-Bandwidth Wall）**：
   对于参数量为 $P$（每参数 $b_w$ bytes）、当前活跃 KV Cache 为 $B_{\text{batch}} \cdot L_{\text{kv}} \cdot d_{\text{kv}}$（每元素 $b_{\text{kv}}$ bytes）的自回归解码，若权重无法驻留在片上缓存，每生成一个时间步的最小字节读取量为：
   $$V_{\text{step}}(B_{\text{batch}}, L_{\text{kv}}) \approx P \cdot b_w + B_{\text{batch}} \cdot L_{\text{kv}} \cdot d_{\text{kv}} \cdot b_{\text{kv}}$$
   给定有效显存/片上存储带宽 $B_{\text{mem}}$，单步延迟的带宽下界为：
   $$T_{\text{step, mem}} \ge \frac{P \cdot b_w + B_{\text{batch}} \cdot L_{\text{kv}} \cdot d_{\text{kv}} \cdot b_{\text{kv}}}{B_{\text{mem}}}$$
2. **降低 $T_{\text{step}}$ 的两条正交路径**：
   - **推理软件栈路径（Fireworks AI / Together AI 在 GPU/HBM 上）**：不改变 HBM3/HBM3e 物理介质，而是通过 **(a) 稀疏/异步 Kernel**（只读取 Top-$k$ 选中的 KV 块或用 Warp-Specialization 隐藏 Softmax/TMA 延迟）、**(b) 硬件感知推测解码树（Speculative Tree）**（在一次权重读取中并行验证 $N_{\text{tree}}$ 个候选 token，将有效摊销步耗时降为 $T_{\text{verify}}(N_{\text{tree}}) / \mathbb{E}[\tau]$）、**(c) 亚二次/常状态混合架构**（用固定维度的 FIR/IIR 状态替代线性增长的 KV Cache）。
   - **新型芯片路径（Groq TSP / Cerebras WSE 在 SRAM 上）**：彻底抛弃片外 HBM，将全部权重与激活放入**片上 SRAM**（Groq 单芯片 `220 MiB` 显式管理 SRAM 达 `55 TiB/s` 内部带宽；Cerebras WSE 单晶圆数十万 PE 各配 `48 KB` 本地 SRAM 并通过 2D Mesh 片上织物通信），直接把分母 $B_{\text{mem}}$ 提升 $10\times\text{–}1000\times$，但代价是**片上容量墙（Capacity Cliff）与多芯片/跨 PE 确定性切分成本**。

| 公司/架构流派 | 核心公开技术栈（已核验源码/论文） | 物理存储层次与带宽定位 | 解决的核心瓶颈 | 架构硬约束与反证边界（Capacity / Latency Cliff） |
|---|---|---|---|---|
| **Fireworks AI**（GPU 推理加速） | `fw-ai/minimax-kernels`（SM100 M3 Block-Sparse Attention）、`fw-ai/benchmark`（Prefix Cache 压测） | NVIDIA Blackwell B200 / Hopper HBM3e + SM Shared Memory (`TMEM/SMEM`) | 长上下文 KV 带宽墙（只读 `num_sel` 个 128-token KV 块）与 Prefix Cache 命中率控制 | 需动态构建 KV-outer 反向索引（5 个 CuTe-DSL 预处理 kernel + 负载均衡调度），稀疏度不足或小序列时索引开销抵消收益 |
| **Together AI**（GPU Kernel + 算法协同） | `FlashAttention-3` (`arXiv:2407.08608`)、`Sequoia` (`arXiv:2402.12374`)、`StripedHyena`、`MoA` | Hopper H100 `WGMMA + TMA` 异步流水线 + HBM3 | Attention 非矩阵算子气泡、Batch-1 权重读取摊销（Speculative Tree）、长序列 KV 容量增长 | Sequoia 树过大时进入 Compute-bound 区导致验证延迟飙升；MoA 多轮多模型扇出使 Prefill/TTFT 与总 Token 成本成倍放大 |
| **Groq**（TSP / LPU 确定性流处理器） | `arXiv:2206.11062`（TSP 微架构）、`groq/groqflow`（Daisy-Chain / Dragonfly / Rotational 多芯片编译） | 单芯片 `220 MiB` SRAM（88 块伪双口 SRAM，内部 `55 TiB/s`，SRF 流 `20 TiB/s`），无片外 DRAM/HBM | 彻底消除 HBM 延迟抖动与硬件 Cache 争用，Batch-1 极低确定性尾延迟（P99 仅比均值高 `0.4%`） | 单芯片仅约容纳 `~50M` 参数（`groqflow` 按 `ceil(P / 5e7)` 2 的幂次分配芯片）；大模型需数十至上百张卡 Daisy-Chain 串联，受机架成本/功耗与芯片间跳数约束 |
| **Cerebras**（WSE 晶圆级数据流阵列） | `Cerebras/sdk-examples`（CSL 语言、2D Mesh Color 路由、片上 AllReduce、SUMMA GEMM/GEMV） | 晶圆级 2D PE 网格，每个 PE 独立 `48 KB` SRAM（单周期访问）+ 片上单周期 Wavelet 交换路由 | 将分布式集群通信转化为晶圆内单颜色（`1 color`）硬件 Wavelet 流与动态开关重构 | 主机与晶圆边缘 I/O（`H2D/D2H` 最多 16 通道 Wavelet 流）远窄于片上带宽；权重与状态必须按 `[M/H, N/W]` 静态切分进 `48 KB/PE` 局部内存 |

---

## 2. Fireworks AI：Blackwell SM100 块稀疏 Attention 与生产级缓存负载压测

### 2.1 `fw-ai/minimax-kernels`：M3 KV-Outer Block-Sparse Attention 的四阶段实现

在固定提交 [`fw-ai/minimax-kernels@b60c3f290f3eb7227d23b50f169e5145d3326288`](https://github.com/fw-ai/minimax-kernels/blob/b60c3f290f3eb7227d23b50f169e5145d3326288/docs/m3-sparse-attention.md)（`2026-07-14`，`implementation`）中，Fireworks 公开了面向 Blackwell SM100（B200）的 **M3 Sparse Attention** 实现。其核心问题在于：在 GQA（例如 `Hq = 64` 个查询头共享 `Hkv = 8` 个 KV 头，每个 KV 组包含 `G = 8` 个查询头）的长上下文场景下，每个 Query 位置通过 Top-$k$ 路由只选择 `num_sel` 个 KV 块（每块 `block_size = 128` tokens）。若按传统 **Q-outer（以 Query 为外循环）** 方式执行，不同 Query 块选中的同一个 KV 块会被反复从 HBM 读入 SM，造成严重的 HBM 读放大。

M3 采用 **KV-outer（以 KV 块为外循环）** 反向索引与双后端（C++ AOT 编译 `.so` 与 Python CuTe-DSL JIT）架构，每次前向传播由 **4 个 GPU 阶段** 串联完成：

1. **Stage 1 — `build_kvouter_index`（稀疏图转置为 CSR 反向索引）**：
   将形状为 `[B, Sq, Hkv, num_sel]` 的 `topk_idx`（每个 Query 选了哪些 KV 块）转置为“每个 `(batch, kv_head, kv_block)` 被哪些 Query token 选中”的压缩稀疏行（CSR）结构。源码通过 **5 个顺序执行的 CuTe-DSL GPU kernel** 实现：
   - `InitSlotsAndCounts`：清零计数表与副本槽位；
   - `CountEdges`：利用 `atomicAdd` 统计落入每个 KV 块的 Query 边数；若单个 KV 块的入度超过 `replica_Threshold = 128`，则分配多个副本（replicas）以拆分热点 KV 块；
   - `ReduceReplicas`：归约各副本计数；
   - `CountToOffsets`：计算前缀和（Prefix Sum）生成 CSR 偏移数组 `offsets`；
   - `ScatterRanks`：利用原子计数器将选中的 Query 位置与路由权重写入连续区间，输出 `unique_kv_list`、`counts`、`offsets` 与 `reorder_idx`。
2. **Stage 2 — `build_load_balanced_schedule`（SM100 持久化 CTA 负载均衡）**：
   单个 Triton/CuTe kernel 采用 Longest-Processing-Time-First（LPT）贪心算法，将非空的 KV 块任务分配到固定数量的 `num_ctas`（在 B200 上设为 `148`，严格对齐 B200 的 `148` 个物理 SM），避免某些 KV 块被大量 Query 选中而导致持久线程块（Persistent CTA）木桶效应。
3. **Stage 3 — `SparseKVOuterForward`（SM100 主计算 Kernel）**：
   每个持久 CTA 遍历其分得的 KV 块：将 `128` 个 token 的 KV 块一次性加载进片上存储，流式遍历所有选中该 KV 块的 Query 块（步长 64），完成 $Q K^T$、在线局部 Softmax 和 $P V$ 累加，并将每个 Query 针对该 KV 块组的局部输出与对数和指数（`partial LSE`）写出。
4. **Stage 4 — `LSE combine`（跨 KV 分区精确归约）**：
   利用对数域稳定公式将同一 Query 在不同 KV 分区上的局部结果合并为精确全局 Attention 输出：
   $$\text{LSE}_{\text{global}} = \log\sum_{r} \exp(\text{LSE}_r), \quad O_{\text{global}} = \sum_{r} \exp(\text{LSE}_r - \text{LSE}_{\text{global}})\, O_r$$

### 2.2 作者 B200 性能数据的严谨口径区分（`author-benchmark`）

在 [`docs/perf-results-b200.md`](https://github.com/fw-ai/minimax-kernels/blob/b60c3f290f3eb7227d23b50f169e5145d3326288/docs/perf-results-b200.md)（`author-benchmark`）中，作者披露了在单卡 NVIDIA B200（`148 SMs`，CUDA 13.0，20 次 warmup + 100 次 CUDA Event 计时）上的评测结果，并显式定义了两套不可混算的带宽口径：
- **`fio1x` / `fbw1x`（Cold-Cache / 理想无重读口径）**：假设全部选中输入与输出张量仅从 HBM 读写 1 次；
- **`fioAct` / `fbwAct`（Actual algorithmic HBM traffic 口径）**：计入 KV-outer 遍历过程中 Query 张量跨越多个稀疏 KV 分区时的多次读取与 partial 写出。
- 在作者报告的 `Sq = Skv = 32768, Hq = 64, Hkv = 8, D = 128` 扫描中，当 `num_sel` 从 `8`（选中 1,024 tokens）增加到 `64`（选中 8,192 tokens）时，C++ 后端中位耗时从 `0.335 ms` 增至 `1.481 ms`，有效算力 `TFLOPs(u)` 从 `458 TFLOP/s` 升至 `691 TFLOP/s`（`num_sel = 32` 时达 `724 TFLOP/s`），`fbw1x` 为 `2.88–3.12 TB/s`，而计入重复访问的 `fbwAct` 达 `5.10–5.75 TB/s`。
- **架构师反证与警示**：若把 `fbwAct`（含片上 L2 命中支撑的重复逻辑访问量）当成 HBM 物理引脚带宽，或者忽略 `Sq` 较短时 Stage 1 五个原子预处理 kernel 的固定启动开销，就会在容量规划中严重高估稀疏 Attention 在短序列或高密度选中率下的端到端加速比。

### 2.3 `fw-ai/benchmark`：Prefix Cache 控制变量与推理负载生成

在 [`fw-ai/benchmark@94cd111967c7be51a35777c475d7c475d2b66ec94a`](https://github.com/fw-ai/benchmark/blob/94cd111967c7be51a35777c475d7c5d2b66ec94a/llm_bench/load_test.py)（`2026-04-16`，`implementation`）中，`llm_bench/load_test.py` 展示了推理平台压测最容易失真的变量控制：
- 若每次请求使用完全相同的 prompt，服务端 Prefix Cache（RadixAttention / Prompt Cache）会 100% 命中，测出的 TTFT 只是缓存读取延迟而非真实 Prefill 算力；
- 该代码通过 `--prompt-cache-max-len` 显式构造 `common_tokens`（共享前缀长度）与每次随机采样的后缀 token 拼接（`TranslationDataset.sample`），并分别上报 `time_to_first_token`（TTFT）、`latency_per_token`（TPOT）、`generation_tokens_per_second` 与 `full_tokens_per_second`。这验证了评估任何商用推理 API 时，**必须显式锁定共享前缀比例（Prefix Cache Hit Ratio）与输出长度分布**。

### 2.4 源码深挖（2026-10-16 新增）：CuTe-DSL 辅助管线如何消除原子争用、D2H 同步与中间归一化开销？

进一阅读固定提交 `fw-ai/minimax-kernels@b60c3f290f3eb7227d23b50f169e5145d3326288` 下的三份核心 CuTe-DSL 源码，可以定位块稀疏 Attention 在 GPU 端落地的三个微观工程瓶颈及其解法：

1. **自适应原子计数私有化与共享内存 Hillis-Steele 扫描（[`build_kvouter_index.py`](https://github.com/fw-ai/minimax-kernels/blob/b60c3f290f3eb7227d23b50f169e5145d3326288/py/minimax_kernels/m3_sparse_attention/build_kvouter_index.py)，`blob: e7edfbf3f279c1e76c0febfdfea997c77b9eda2d`）**：
   - **问题**：当 `Tq` 很大但 KV 块槽位数 `nbins = Hkv * num_block_slots` 较少（或大量 Query 集中命中热点 Attention Sink 块）时，`edges = Tq * Hkv * topK` 条边对同一个全局 `mCount[h_kv, edge_slot]` 执行 `atomic_add` 会引发严重的 L2 原子串行化。
   - **源码实现（`L34–59, L266–348, L416–453`）**：`_adaptive_replicas(edges, nbins)` 根据平均争用度动态选取 2 的幂次副本数 $R \in [16, 128]$，使每个副本计数器的期望边数贴近 `_COUNT_TARGET_CONTENTION = 4`：
     $$R = \text{ clamp}_{[16, 128]}\!\left(2^{\lceil \log_2(\text{edges} / (\text{nbins} \times 4)) \rceil}\right)$$
   - 在 `_CountEdgesKernel` 中，CTA `bidx` 将原子加分散写入 `mCount[h_kv, edge_slot, bidx % R]`（`sem="relaxed", scope="gpu"`）；紧接着 `_ReduceReplicasKernel` 启动 **`grid = [Hkv * num_block_slots, 1, 1]`（每个 `(hkv, slot)` 独占一个 CTA）**，在 Shared Memory 中对 $R$ 个副本执行 $\log_2 R$ 步 **Hillis-Steele 闭区间前缀扫描**，把每个副本的原值原地改写为其槽位内排他偏移基址（`mCount[h, slot, tidx] = Int32(s[tidx]) - own`），由 `_ScatterRanksKernel` 在一次无锁写入中同时生成 CSR 索引 `kv_to_q_indices_and_ranks` 与供 Combine 收集使用的逆映射表 `inv[h_kv, t, rank] = out`。
   - **未分配物理页安全重映射（`L145–159`）**：在 Paged KV 模式下，块表未分配槽位填充为 `-1`。若直接把 `-1` 钳制到物理页 `0`，会错误读取共享池中属于其他请求的活跃第 0 页；源码显式回退到**当前序列自身的第 0 个已分配物理页 `mBlockTables[seq_id, 0]`**，再由因果掩码与 `used_kv_lens` 屏蔽越界行。

2. **零 D2H 同步的单 Kernel 负载均衡器（[`sparse_fwd_kvouter_load_balance_schedule.py`](https://github.com/fw-ai/minimax-kernels/blob/b60c3f290f3eb7227d23b50f169e5145d3326288/py/minimax_kernels/m3_sparse_attention/sparse_fwd_kvouter_load_balance_schedule.py)，`blob: 2f1c47a1a44b2dfbb4a8d918ca9f579314f8e585`）**：
   - 早期 PyTorch 实现需要 `diff + cumsum + arange + 2x searchsorted + gather + where` 共约 **8 次 Kernel 启动**并物化临时张量。
   - `_LoadBalanceScheduler`（`L47–142`）利用 `kv_to_q_offsets[h]` 本身已是单 KV 头内单调前缀和的性质，在 Host 侧按最大容量上界计算每 Split 配额 $\text{nqps} = \lceil (H_{kv} \cdot T_q \cdot \text{topK}) / \text{num\_splits} \rceil$（无需任何 Device$\to$Host 拷贝），仅启动 **1 个 CuTe Kernel**：每个线程对应一个 Split $s$，先用编译期展开的 $H_{kv}$ 步扫描定位 `head`，再用固定 `_SEARCH_ITERS = 32` 步幂等二分查找定位 `(blk, q)` 起点与终点，复杂度 strictly $O(H_{kv} + \log_2 \text{nbs})$。

3. **Tile-Ordered 逆映射收集与 `exp2` 延迟归一化合并（[`sparse_fwd_kvouter_combine.py`](https://github.com/fw-ai/minimax-kernels/blob/b60c3f290f3eb7227d23b50f169e5145d3326288/py/minimax_kernels/m3_sparse_attention/sparse_fwd_kvouter_combine.py)，`blob: 4d925a73482d27b717cef54497e17f2d6ccaebe5`）**：
   - 在 Flat（Tile-Ordered）模式下，主计算 Kernel 直接按 KV-outer 顺序紧密写出**未归一化（RAW）**的局部累加器 `O_partial`（形状 `[Hkv * Tq * topK * qhead, D]`）、以 2 为底的最大值 $\tilde{m}_s = \text{row\_max} \cdot \log_2 e$ 以及局部指数和 $l_s$，省去了主 Kernel 末尾的逐行除法。
   - `FlashAttentionForwardCombine`（`L223–548`）通过 `p = inv[hkv, q, split]` 间接寻址每个 Query 的 `topK` 个局部结果（若 `p < 0` 表示该边被因果掩码丢弃，则向 Shared Memory 填 `0` 以避免 `NaN`），并以 Token-Major 顺序（`idx = q * num_head + head`）排布线程，使同一 Query 的 `qhead` 个 GQA 查询头在 `(hkv * G + p) * qhead + hl` 上命中**连续全局内存行**。
   - 合并权重在单趟 FP32 归约中同时完成跨 Split 合并与最终 Softmax 归一化（`L424–446`）：
     $$a_s = 2^{\tilde{m}_s - \tilde{M}}, \quad D = \sum_{s=0}^{\text{topK}-1} a_s \cdot l_s, \quad O = \sum_{s=0}^{\text{topK}-1} \left(\frac{a_s}{D}\right) O_{\text{partial}, s}, \quad \text{LSE} = \ln(D) + \tilde{M}\ln 2$$

---

## 3. Together AI：Hopper 异步 Kernel、硬件感知推测解码树与常状态模型

Together AI 的公开研究布局覆盖了从底层 CUDA/Cutlass Kernel 到解码算法与模型结构的三层协同：

### 3.1 `FlashAttention-3`（`arXiv:2407.08608`）：用 Warp-Specialization 与乒乓流水线打破算子串行墙

在论文 [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/abs/2407.08608)（`2024-07-12`，`specification` + `author-benchmark`）中，作者指出在 Hopper H100 上，FP16 Tensor Core 峰值（`989 TFLOP/s`）远高于超越函数单元（MUFU，每 SM 仅 `16 ops/clock` 计算 $\exp2$）：
1. **Producer-Consumer Warp-Specialization**：将 CTA 内的 Warpgroups 拆分为专门负责 `TMA`（Tensor Memory Accelerator）异步拷贝 HBM$\to$SMEM 的 Producer Warpgroup，以及专门执行异步 `WGMMA` 与 Softmax 的 Consumer Warpgroups，并通过 `setmaxnreg` 动态重分配寄存器（减少 Producer 寄存器配额、扩大 Consumer 寄存器配额）。
2. **2-Stage GEMM-Softmax 跨迭代重叠**：利用 `bar.sync` 协调两个 Consumer Warpgroups 乒乓执行（当 Warpgroup 1 执行 `WGMMA` 时，Warpgroup 2 独占 MUFU 执行 Softmax），并在同一 Warpgroup 内把当前块 $i$ 的 $\tilde{P}_i V_i$（GEMM1）与下一块 $i+1$ 的 $Q K_{i+1}^T$（GEMM0）重叠在 Softmax 之前发射。
3. **FP8 块量化与非相干处理（Incoherent Processing）**：用随机正交矩阵（Hadamard 变换）乘以 $Q, K$ 以打散通道离群值（outliers），配合每块独立缩放因子。作者报告在 H100 PCIe/SXM 上 FP16 前向达到最高 `740 TFLOP/s`（约 `75%` 利用率），FP8 达到约 `1.2 PFLOP/s`（`author-benchmark`）。

### 3.2 `Sequoia`（`arXiv:2402.12374`）：硬件感知的动态规划推测解码树

在 [*Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding*](https://arxiv.org/abs/2402.12374)（Together AI & CMU，`specification` + `author-benchmark`）中，作者形式化了推测解码（Speculative Decoding）的树拓扑优化问题：
- **预期生成长度公式**：设草稿树 $\mathcal{T}$ 含 $N$ 个节点（总树预算），节点 $v$ 的深度为 $d(v)$，从根到 $v$ 沿途每次在草稿分布中采样第 $k_e$ 个分支并被目标模型接受的条件概率乘积为 $P_{\mathcal{T}}(v)$，则一次目标模型并行验证生成的期望 token 数为：
  $$F(\mathcal{T}) = \sum_{v \in \mathcal{T}} P_{\mathcal{T}}(v)$$
- **硬件感知（Hardware-Aware）树大小拐点**：在目标 GPU 上并行验证一棵含 $N$ 个节点的推测树时，由于当 $N$ 较小时目标模型前向传播完全处于 **Memory-bandwidth bound** 区，验证耗时 $T_{\text{verify}}(N) \approx T_{\text{verify}}(1)$ 几乎不随 $N$ 增长；但当 $N$ 超过目标 GPU 的算术强度拐点（Roofline Ridge Point）后，$T_{\text{verify}}(N)$ 开始随 $N$ 显著上升。因此真实加速比由下式决定：
  $$\text{Speedup}(\mathcal{T}) = \frac{F(\mathcal{T})}{\frac{T_{\text{verify}}(|\mathcal{T}|)}{T_{\text{target}}(1)} + d(\mathcal{T}) \cdot \frac{T_{\text{draft}}(1)}{T_{\text{target}}(1)}}$$
  Sequoia 通过测量目标硬件的 $T_{\text{verify}}(N)$ 阶梯曲线，联合动态规划（`sequoia_tree_construction`）搜索最优 $(N, d)$ 对。
- **源码深挖（2026-10-17 新增，`Infini-AI-Lab/Sequoia@0ff3ff71475ca148f2dfb7545e0de5caf5c67b39`）**：
  阅读 [`tree_search.py`](https://github.com/Infini-AI-Lab/Sequoia/blob/0ff3ff71475ca148f2dfb7545e0de5caf5c67b39/tree_search.py)（`blob: e717625933b1abb00bc4c63934ac3564593ee06c`，`L25–128`）与 [`Tree/SpecTree.py`](https://github.com/Infini-AI-Lab/Sequoia/blob/0ff3ff71475ca148f2dfb7545e0de5caf5c67b39/Tree/SpecTree.py)（`blob: 5cf0d15cae6d019ade4fddd5328460877ba1dc44`，`L41–228`）揭示了 Sequoia 树在 GPU 上的精确构建与零冗余 KV 回收机制：
  1. **三维动态规划递推方程（`dynamic_program`, `tree_search.py:25–54`）**：维护状态张量 $T[m, l, b]$（预算 $m$ 个节点、最大深度 $l$、根节点分支度 $b$ 的最大期望接受长度）与回溯表 `branch_map`：
     $$T[m, l, 1] = 1 + \alpha_1 \max_{b'} T[m-1, l-1, b'], \qquad T[m, l, b] = \max_{1 \le y < m} \left( T[y, l, b-1] + \alpha_b \max_{b'} T[m-y, l-1, b'] \right)$$
     其中 $\alpha_b$ 是父节点前 $b-1$ 个候选分支均被拒绝后、第 $b$ 个无放回采样分支被接受的边际概率；随后 `get_optimal_tree_size_and_depth`（`L57–76`）在实测硬件耗时代价 $c = T_{\text{draft}}(1)/T_{\text{target}}(1)$ 与 $t_i = T_{\text{verify}}(b_i)/T_{\text{target}}(1)$ 下最大化 $F(b_i, d) / (c \cdot d + t_i)$。
  2. **扁平化树掩码与位置编码嵌入（`get_grow_map` & `SpecTree.__init__`, `SpecTree.py:45–65`）**：每个子节点继承父节点的注意力掩码行并点亮自身对角线（`attention_mask[i] = attention_mask[parents[i]]; attention_mask[i, i] = 1`），在送入目标模型前把 `0` 项填充为 `torch.finfo(dtype).min`，并将深度直接映射为位置编码偏移 `position_ids = grow_map["depth"][1:] + len(prefix) - 1`，从而把整棵不规则投机树转化为单次长度为 `tree_size` 的 1D 序列前向传播。
  3. **残差分布无放回拒绝采样与增量 KV Gather（`SpecTree.accept_step` & `verify`, `SpecTree.py:137–228`）**：验证时对父节点下的候选子节点依次检验 $p[\text{token}] > r \cdot q[\text{token}]$；一旦某个分支被拒绝，立即调用 `self.residual_graph(p, q)` 更新目标残差分布并将草稿分布对应位置置为 `finfo.min`（`L155–156`）。验证结束后，仅对被接受路径 `accept_list` 调用 `kv_cache.gather_kv_incremental(accept_list[ground_truth_len:], ground_truth_len)`（`L226–227`），将树中未命中的旁支 KV 槽位原地覆写回收，完全避免了树分支带来的 KV Cache 碎片或完整重算。

### 3.3 `StripedHyena` 与 `MoA`：模型状态与系统扇出的两个极端

1. **`togethercomputer/stripedhyena@7e13f618027fea9625be1f2d2d94f9a361f6bd02`**（[`stripedhyena/engine.py:39-227`](https://github.com/togethercomputer/stripedhyena/blob/7e13f618027fea9625be1f2d2d94f9a361f6bd02/stripedhyena/engine.py#L39-L227)，`implementation`）：
   展示了混合状态空间/卷积架构在自回归解码阶段如何消除 KV Cache 线性增长：
   - Prefill 阶段提供两条路径：`prefill_via_modal_fft`（通过 `flashfftconv` 或 FFT 一步算出最终极点状态 `iir_state`）与省显存的 `prefill_via_direct_recurrence`；
   - Decode 阶段（`step_fir` 与 `step_iir`）：短滤波器（FIR）仅维护长度为 `fir_length - 1` 的滑动窗口，长滤波器（IIR）按一阶复数极点递推更新常维度状态：
     $$\text{state}_{t} = \text{poles} \odot \text{state}_{t-1} + x_t$$
     其单层状态字节数对序列长度 $L$ 为 $O(1)$ 常数，不会像标准 Transformer 那样随 $L$ 线性耗尽 HBM。
2. **`togethercomputer/MoA@1b5cab0f0905d9da821e37322ac6df96ba65e1a7`**（[`README.md`](https://github.com/togethercomputer/MoA/blob/1b5cab0f0905d9da821e37322ac6df96ba65e1a7/README.md)，`implementation`）：
   Mixture-of-Agents 将 $M$ 个 Proposer 模型分 $L_{\text{layer}}$ 层串联，每层将前一层全部 $M$ 个模型的完整输出拼接进下一层每个模型的上下文，最后由 Aggregator 汇总。架构代价非常直观：单次用户查询触发 $(L_{\text{layer}} - 1)\cdot M + 1$ 次模型生成，且第 $l$ 层的输入上下文长度按 $O(M \cdot L_{\text{out}})$ 膨胀，端到端 TTFT 取决于每一层最慢 Proposer 的完成时间（$\sum_{l} \max_m T_{l,m}$）。

---

## 4. Groq：TSP/LPU 全片上 SRAM 确定性流处理器与多芯片 Daisy-Chain 编译

### 4.1 TSP 微架构（`arXiv:2206.11062`）：为什么 P99 尾延迟能贴近均值？

在 Groq 团队发表的论文 [*Answer Fast: Accelerating BERT on the Tensor Streaming Processor*](https://arxiv.org/abs/2206.11062)（`2022-06-22`，`specification` + `author-benchmark`）中，第 II-B 至 IV-C 节详细披露了第一代 **Tensor Streaming Processor (TSP)** 的物理指标与确定性原理：

1. **空间功能布局与 320-byte 物理向量流**：
   - 芯片面积 `725 mm²`（`14 nm` 工艺，`26.8B` 晶体管，TDP `275 W`，文中运行频率 `900 MHz`）。
   - 全芯片无硬件指令缓存、无数据 Cache 层次、无片外 HBM/DRAM，而是配置 **`220 MiB` 软件显式寻址的片上 SRAM（MEM）**，分为东西两半球共 `88` 个独立伪双口 SRAM 块（支持同时读写相反 bank，峰值片上 SRAM 带宽 **`55 TiB/s`**）。
   - 算子单元围绕 `45` 个流寄存器文件（SRF）水平排列，由 `64` 条硬件数据流（32 东向、32 西向）以每周期 1 个 320-byte 向量的速率在相邻 SRF 间流式传输（流带宽 **`20 TiB/s`**）：
     - **MXM（Matrix Execution Module）**：4 个独立的 `320 × 320` 2D 乘累加平面（支持 `int8` 与 `fp16`），每个平面每周期对已装载的 `320 × 320` 权重矩阵与流式输入的向量完成一次矩阵乘并输出 320 元 `int32` 向量；当矩阵大于 `320 × 320` 需要多趟（passes）时，硬件在执行第 $i-1$ 趟的同时后台装载第 $i$ 趟权重，消除 MXM 空泡。
     - **SXM（Switching Execution Module）**：位于 MEM 与 MXM 之间，在数据从 SRAM 流向 MXM 的途中直接完成 `distribute / mask / transpose / rotate / permute / shift`（例如 Multi-Head Attention 分头无需把转置结果写回内存）。
     - **VXM（Vector Execution Module）**：位于芯片中心（与 MEM 边界带宽 `40 TiB/s`），含 **16 个可串联（chaining）的 ALU**。文中用 13 个 ALU 串联计算 `GELU`、3 个 ALU 串联完成 `int32 -> fp32` 反量化与 `fp32 -> int8` 重量化，使 `GELU` 直接流水线重叠在 `GEMM` 输出流之后；LayerNorm 则用 4 条并行的 4-ALU 链在 3 趟扫描内完成，周期公式为：
       $$\text{LN\_cycles} = 3 \cdot j \cdot \left\lceil\frac{k}{320}\right\rceil \cdot \frac{1}{4} + c$$
2. **确定性尾延迟（Deterministic Tail Latency）的证据与唯一抖动源**：
   - 作者在编译前根据静态指令周期预测的延迟与实机硬件测量值误差在 **`1%` 以内**。
   - 在 4,000 次 `batch=1, seq_len=128` 的量化 BERT-base 端到端实测（含 Host$\leftrightarrow$TSP 拷贝，`author-benchmark`）中，平均延迟为 `128.9 µs`（标准差 `3.8 µs`），**P1 为 `128.5 µs`，P99 为 `129.5 µs`（P99 仅比均值高 `0.4%`）**，对比同期 A100 TensorRT 的均值 `630 µs`、P99 `790 µs`（因 GPU 缓存争用导致 P99 跳升 `25%+`）。
   - **唯一尾延迟波动源**：论文图 10 证明，片上执行时间在 P1 与 P99 完全恒定，唯一的微秒级波动来自 **Host 与 TSP 之间 16-lane PCIe Gen4 链路的数据包传输错误重放（PCIe replay）**。

### 4.2 `groq/groqflow`：单芯片 `~50M` 参数容量墙与多芯片 Daisy-Chain 拓扑

当模型从 `110M` 参数的 BERT 扩展到数十亿参数的 LLM 时，`220 MiB` 片上 SRAM 成为硬约束。固定提交 [`groq/groqflow@720e024414be401a7853931a2622a26446883746`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/common/build.py)（`2023-12-19`，`implementation`）暴露了真实的工程切分规则：
- **芯片数量估算公式（[`groqflow/common/build.py:92-123`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/common/build.py#L92-L123)）**：
  `calculate_num_chips` 按每张芯片可容纳 **`50,000,000`（50M）个参数** 计算最小芯片数，并向上取整为 2 的幂次：
  $$N_{\text{chips}} = 2^{\left\lceil \log_2 \lceil P / 50{,}000{,}000 \rceil \right\rceil}$$
  并定义了 `DRAGONFLY`（`DF_A14_2/4/8/16/32/64_CHIP`）与 `ROTATIONAL`（`RT09_A14_16/32/40/48/56/64/72_CHIP`）两类基于 `A1.4` 卡的多芯片物理拓扑。
- **Daisy-Chain 分区编译与大程序指令切片借用**：
  - [`groqflow/justgroqit/compile.py:139-145`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/justgroqit/compile.py#L139-L145) 调用 `groq-compiler` 时传入 `--multichip={state.topology}` 与 `--partition-mode=daisy-chain`，将模型按层/子图切分为 $N_{\text{chips}}$ 份独立的 Alan Assembly（`.aa`）文件；
  - [`groqflow/justgroqit/assemble_multichip.py:67-115`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/justgroqit/assemble_multichip.py#L67-L115) 在开启 `--large-program` 时，通过 `--extra-ifetch-slices` 征用原本用于数据的 SRAM 切片（`West 18,19` 与 `East 17,18,19,38`）存放指令，证明**超长静态调度程序会直接挤占片上常量/权重 SRAM 空间**。
  - [`groqflow/justgroqit/ignition.py:129-205`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/justgroqit/ignition.py#L129-L205)（2026-10-17 补充核验，`blob: afc4f1f53dcf89c6bce3debb51d8c597f6bd5ba5`）进一步硬编码了单芯片默认汇编器标志 `--ifetch-from-self` 与 `--ifetch-slice-ordering=round-robin`，并在 `_validate_args`（`L199–204`）中强制规定**自定义 `assembler_flags` 与 `num_chips != 1` 互斥**（多芯片构建完全由编译器统一接管跨卡同步与指令切片分配）。
  - **多芯片 C2C 拓扑初始化、$N_{\text{chips}}$ 步流水排空与分片 `.iop` 缓存校验（[`groqflow/groqmodel/execute.py:15-101`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/groqmodel/execute.py#L15-L101), `blob: 4e9482e801721f27cfb67f8343d2181d3afdaf5e`；[`groqflow/groqmodel/remote.py:71-150`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/groqmodel/remote.py#L71-L150), `blob: b140b67305783e802c3cefd1f2d8d348c0748fdd`）**：
    - 在集群/机架侧拉起多卡任务时，`get_multi_tsp_runner`（`execute.py:15–42`）在首次运行或物理链路重配后必须调用 `tsp.bringup_topology(user_config=topo_config[topology])`（如 `DF_A14_2/4/8_CHIP`）完成 **Chip-to-Chip (C2C) 确定性同步链路 Bring-up**；
    - 更关键的是，`run()` 中的 `forward_multichip(example)`（`execute.py:77–84`）明确揭示了 `MultichipTSPRunner` 的流水线推进语义：单次 `tsp_runner(**example)` 仅将输入推进 **1 张芯片（1 pipeline stage）**，因此对单样本冷启动推理，必须连续调用 **`for _ in range(num_chips): output = tsp_runner(**example)`** 共 $N_{\text{chips}}$ 次才能让张量走完整个 Daisy-Chain 图！
    - 在远端执行编排器 `RemoteGroqModel`（`remote.py:71–150`）中，单芯片构建生成单个 `output.iop`，而 $N_{\text{chips}} > 1$ 时生成 `N` 个独立的芯片级二进制镜像 `output.0.iop ... output.{N-1}.iop`；调度器在每次执行前通过 `_calc_iop_md5s()` 计算全部 $N$ 个 `.iop` 文件的 MD5 摘要数组并请求 `/storage/cache/check`，唯有全量命中远端芯片缓存（`cache_hit == True`）才跳过多卡二进制分发（`skip_iop_urls=True`）。

---

## 5. Cerebras：WSE2/WSE3 晶圆级 2D Mesh、单颜色片上 AllReduce 与边缘 I/O 边界

在固定提交 [`Cerebras/sdk-examples@d19f2ca3e51285fc5904c78643b38819a2f26c44`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/README.rst)（`2026-04-27`，`implementation`）中，Cerebras Software Language（CSL）代码展示了 Wafer-Scale Engine（WSE2/WSE3）如何将芯片设计与分布式通信合二为一：

### 5.1 每个 PE `48 KB` 本地 SRAM 与单颜色（`1 Color`）2D Mesh 片上 AllReduce

在 [`benchmarks/benchmark-libs/allreduce/pe.csl:1-220`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/benchmark-libs/allreduce/pe.csl#L1-L220) 与 [`layout.csl`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/benchmark-libs/allreduce/layout.csl) 中：
- **严苛的每核心资源预算**：每个 Processing Element（PE）仅有 **`48 KB` 本地 SRAM**（在单向量极限测试中最多分配 `4096` 个 `f32` 元素 = `16 KB`），且全晶圆 AllReduce 库仅占用 **1 个可路由颜色（routable color `C_ROUTE`）、1 个输入队列（`iq_route`）、1 个输出队列（`oq_route`）和 3 个数据流寄存器（DSR：`dest_dsr, src0_dsr, src1_dsr`）**。
- **硬件 Wavelet 动态开关重构（Dynamic Switch Reconfiguration）**：
  片上互连不使用软件 TCP/RDMA 协议栈，而是通过 32-bit 数据/控制小波（Wavelet）驱动每个 PE 的硬件路由器（Switch）。AllReduce 状态机分为四步：`STATE_ROW_REDUCE`（沿西$\to$东归约到最右列 PE）$\to$ `STATE_COL_REDUCE`（最右列沿北$\to$南归约到右下角 PE `(W-1, H-1)`）$\to$ `STATE_BCAST`（从右下角沿北、西两向广播回全网格）$\to$ `STATE_DONE`。
  - 在归约阶段，上游 PE 发送完自身向量后，向输出队列注入带控制位的 Wavelet（`ctrl_11`，含 `OPCODE_SWITCH_ADV`），硬件开关收到后自动把路由从“发送本地 PE（`RAMP`）”切换到“透传上游方向（`WEST/NORTH`）”；
  - 最后一个片段发送拆卸小波（Teardown Wavelet，`0x9df9249`），触发全路径 PE 的 `@get_config_unchecked` 中断处理函数，在原地改写路由器配置寄存器以进入下一阶段（例如从 `ROW_REDUCE` 切换为 `COL_REDUCE` 或 `BCAST`），并直接用 `@fadds(mem_dsr, mem_dsr, fab_in_dsr)` 将网络队列流入的 Wavelet 与本地 SRAM 向量做单指令流式相加。

### 5.2 2D 分块矩阵乘（SUMMA GEMM / GEMV）与主机边缘 I/O 带宽计时

1. **分布式片上 GEMM / GEMV（[`benchmarks/gemm-collectives_2d/README.rst`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/gemm-collectives_2d/README.rst)、[`benchmarks/gemv-collectives_2d/README.rst`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/gemv-collectives_2d/README.rst)）**：
   因为单个 PE 只有 `48 KB` SRAM，矩阵 $A \in \mathbb{R}^{M \times K}$、$B \in \mathbb{R}^{K \times N}$ 必须静态切分为 $\frac{M}{H} \times \frac{K}{W}$ 等分驻留在 $H \times W$ 个 PE 上，通过 `collectives_2d` 库执行 $P$ 步 SUMMA 行广播与列广播累加。
2. **主机到晶圆（`H2D` / `D2H`）边缘带宽与硬件时钟（[`benchmarks/bandwidth-test/README.rst`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/bandwidth-test/README.rst)、[`src/bw_sync_layout.csl`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/bandwidth-test/src/bw_sync_layout.csl)）**：
   - 晶圆内部 PE 间具备极高的 2D Mesh 带宽，但主机与晶圆之间的数据注入/流出（`memcpy_h2d` / `memcpy_d2h`）受限于晶圆边缘的流式 I/O 通道数（代码中约束 `1 <= --channels <= 16`）。
   - 代码通过片上 48-bit 硬件时间戳计数器（TSC，按固定 **`0.85 GHz`** 换算：`time_us = (cycles / 0.85) * 1e-3`，每个 Wavelet 为 `4 bytes`）测量主机流式吞吐：
     $$B_{\text{io}}\text{ (MB/s)} = \frac{N_{\text{wavelets}} \times 4\text{ bytes}}{T_{\text{send/recv}}\text{ (µs)}} \times N_{\text{loops}}$$
   - **全晶圆硬件屏障同步与 WSE-2 vs WSE-3 队列颜色绑定差异（[`benchmarks/bandwidth-test/src/sync/pe.csl:80-307`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/bandwidth-test/src/sync/pe.csl#L80-L307)，`blob: dc35b1ed763f25ffd73d2556390a4637726d6bc8`）**：
     - 因为晶圆上数十万个 PE 各自拥有独立的本地 48-bit TSC 计数器，在测量微秒级 H2D/D2H 带宽前，必须将全晶圆 PE 的参考时钟 $T_0$ 对齐。`sync/pe.csl` 实现了一个**零主控开销的三阶段奇偶双色硬件屏障**：
       1. **`f_sync()`（行向东$\to$西归约）**：每行用两个颜色 `C0, C1` 奇偶交替接力（`P_even` 收 `C0` 发 `C1`，`P_odd` 收 `C1` 发 `C0`，避免同色直通合并），从最右列 `px = width - 1` 逐跳传至最左列 `px = 0`（`L128–142, L217–243`）；
       2. **`f_sync_y()`（首列南$\to$北归约）**：仅 `px = 0` 的第一列用颜色 `C2, C3` 奇偶交替从底行 `py = height - 1` 向上汇聚至左上角根核心 `P(0,0)`，其余列直接挂起等待广播（`L149–168, L245–270`）；
       3. **`f_sync_bcast()` $\to$ `f_exit()`（单色全网广播锁存 TSC）**：一旦 `P(0,0)` 收到全网格汇聚信号，立即向 `c_bcast` 注入 1 个 Wavelet，沿首列南向（`tx = {EAST, RAMP, SOUTH}`）与各行东向（`tx = {EAST, RAMP}`）硬件多播树广播至全晶圆，所有 PE 在收到该 Wavelet 的瞬间激活 `f_exit()` 执行 `timestamp.get_timestamp(&tscRefBuffer)` 记录 $T_0$ 并调用 `sys_mod.unblock_cmd_stream()` 解锁主机命令流（`L172–191, L273–307`）。
     - 同时，`sync/pe.csl:88–123, 206–214` 展示了 **WSE-2 与 WSE-3 的硬件微架构演进**：在 WSE-2 上，`fabric_color` 绑定在输出数据流描述符 `fabout_dsd` 上；而在 **WSE-3（`@is_arch("wse3")`）** 上，`fabout_dsd` 仅指定 `.output_queue`，颜色绑定下沉到编译期队列初始化 `@initialize_queue(oq, .{ .color = c_send })`，解耦了数据流描述符与物理路由颜色。
   - **架构含义**：Cerebras WSE 只有在**权重常驻晶圆片上 SRAM、避免频繁从主机流式换入权重**时才能发挥晶圆级带宽优势；一旦模型权重超出单晶圆总 SRAM 容量需要频繁跨晶圆/主机流式加载，系统吞吐会立即退化到边缘 I/O 通道上限。

### 5.3 源码深挖（2026-10-16 新增）：`gemv-collectives_2d/pe.csl` 揭示的晶圆级 Batch-1 Decode 通信拓扑

对比 Cerebras SDK 中的 2D 矩阵乘（`gemm-collectives_2d`）与 2D 矩阵-向量乘（[`benchmarks/gemv-collectives_2d/pe.csl`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/gemv-collectives_2d/pe.csl)，`blob: 64a1c2cf9ff725984254043bf6e3709ba3a90a15`），可以清晰看到晶圆级芯片在 **Batch-1 自回归 Decode（$y = Ax + b$）** 下的真实执行链路：
- 在 $P_w \times P_h$ 的 PE 网格上，权重矩阵 $A \in \mathbb{R}^{(M_t P_h) \times (N_t P_w)}$ 被静态切分为每个 PE 本地 SRAM 中的 `A_tile`（`[Mt * Nt]f32`），**推理全过程权重 `A_tile` 零移动**。
- 单次 GEMV 由 5 个硬件任务（`local_task_id` 9–13）通过 `collectives_2d`（`mpi_x` 行通信域与 `mpi_y` 列通信域）串联触发：
  1. **`main()`（左列 Scatter $b$）**：仅 `px == 0` 的最左列参与 `mpi_y.scatter(0, &b_src, &b_tile, Mt, scatter_x_task_id)`，将偏置向量 $b$ 从 `PE(0,0)` 切分下发至左列各行；其余列直接激活 `scatter_x_task_id`。
  2. **`scatter_x()`（顶行 Scatter $x$）**：仅 `py == 0` 的最顶行参与 `mpi_x.scatter(0, &x_src, &x_tile, Nt, broadcast_x_down_task_id)`，将输入激活向量 $x$ 从 `PE(0,0)` 沿水平方向切分至顶行各列。
  3. **`broadcast_x_down()`（全列 Broadcast $x_{\text{tile}}$）**：每列以顶行 `py = 0` 为根执行 `mpi_y.broadcast(0, &x_tile, Nt, compute_task_id)`，使同一列的所有 PE 获得相同的 $N_t$ 维输入向量分片。
  4. **`compute()`（本地 `fmacs` + 行向 `reduce_fadds`）**：每个 PE 循环 $j \in [0, N_t)$ 执行单指令流式乘加 `@fmacs(dsd_local_prod, dsd_local_prod, dsd_A_offset, x_tile[j])`，最左列 `px == 0` 叠加偏置 `@fadds(dsd_local_prod, dsd_local_prod, dsd_b_tile)`，随后整行向最右列 `Pw - 1` 发起浮点归约：**`mpi_x.reduce_fadds(Pw - 1, &local_prod, &row_sum, Mt, gather_result_task_id)`**。
  5. **`gather_result()`（最右列 Gather $y$）**：最右列执行 `mpi_y.gather(Ph - 1, &row_sum, &final_result, Mt, EXIT)`，将完整输出向量汇聚至右下角 `PE(Pw-1, Ph-1)`。
- **架构师核心推论**：在晶圆级 2D Mesh 上，Batch-1 Decode 的本地计算仅有 $N_t$ 条向量指令（`@fmacs` 耗时极短），而激活向量需要在每一层经历 **`scatter_x`（$P_w$ 跳）$\to$ `broadcast_y`（$P_h$ 跳）$\to$ `reduce_fadds_x`（$P_w$ 跳）** 的横纵接力。对比 [`benchmarks/gemm-collectives_2d/pe.csl`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/gemm-collectives_2d/pe.csl)（2026-10-17 补充核验，`blob: 4f44a0fb9ff03f4707aeb6884d4d741aa8343c0c`，`L79–133`）的 2D SUMMA 算法——在 $P \times P$ 网格上分 $P$ 步同时执行 `mpi_x.broadcast(step, Ap, Mt*Kt)` 与 `mpi_y.broadcast(step, Bp, Kt*Nt)`，只有当 `x_done()`（`@activate(compute_task_id)`）与 `y_done()`（`@unblock(compute_task_id)`）双双完成时才执行 $K_t \times N_t$ 次本地 `@fmacs`——可以看到：**GEMM（Prefill/训练）中每步双广播携带 $O(M_t K_t + K_t N_t)$ 字节并分摊给 $O(M_t K_t N_t)$ 次乘加，算术强度随分块尺寸增长；而 GEMV（Batch-1 Decode）每层无法在 $N_t$ 维上分摊路由延迟，单 Token 最低延迟下界由 2D Mesh 横纵集体原语的跨 PE 路由跳数 $\propto (P_w + P_h)$ 决定**。

---

## 6. 架构师选型矩阵、计算练习与后续可证伪问题

### 6.1 面向云解决方案架构师的工作负载选型框架

| 维度 | GPU + 现代推理栈（Fireworks / Together 路线） | 全片上 SRAM 确定性流处理器（Groq TSP/LPU 路线） | 晶圆级 2D Mesh 数据流引擎（Cerebras WSE 路线） |
|---|---|---|---|
| **最适配场景** | 高并发混合负载、长上下文（32K–128K+，配合 M3 稀疏/Prefix Cache）、多 LoRA/多模型动态混部 | 对单序列 TPOT / P99 尾延迟有极致要求的实时 Agent、语音对话、代码补全或高频在线决策 | 超高单流解码速度、权重可常驻晶圆网格的密集/结构化稀疏生成与长序列流式计算 |
| **不适用/退化反例** | 极低并发（Batch=1）且未开启 Speculative Decoding 时受限于单卡 HBM 带宽（约 `3.35–8 TB/s`） | 超长上下文（KV Cache 直接挤占宝贵的 `220 MiB/chip` SRAM）或需要频繁动态切换数十个基础大模型 | 算子不规则、高度依赖动态控制流或频繁 Host$\leftrightarrow$Wafer 大块权重换入换出的负载 |
| **容错与运维复杂度** | 成熟的 Kubernetes/vLLM/SGLang 生态，单节点故障可由副本负载均衡摘除 | 强依赖编译器静态指令排布；`N` 芯片 Daisy-Chain 中单张卡故障会导致整条模型流水线中断 | 需专用 CSL / Wafer 编译栈与晶圆级供电散热基础设施，通常以托管云服务或整机柜交付 |

### 6.2 架构师计算与设计练习

1. **容量与芯片数对比计算**：假设某 `8B` 参数模型采用 `INT8` 权重（`1 byte/param`）。
   - 在单卡 `80 GiB` HBM3（有效带宽 `3.0 TB/s`）GPU 上，Batch=1 单步读取全部权重的理论最小耗时是多少毫秒？
   - 按 `groqflow` 的 `calculate_num_chips` 规则（每芯片 `50M` 参数容量并向上取 2 的幂次），部署该 `8B` 模型需要多少张 TSP 芯片组成 Daisy-Chain？若每张芯片片上 SRAM 为 `220 MiB`，16 张或 32 张芯片的总 SRAM 容量分别是多少 `GiB`？为什么不能把 `220 MiB` 全部塞满模型权重（结合论文图 11 的 Scratchpad + Instruction + 长序列 KV 空间回答）？
2. **Speculative Decoding 拐点分析**：根据 Sequoia 的加速比公式，为什么当目标 GPU 已经处于高 Batch 饱和状态（Compute-bound）时，继续开启大树宽的 Speculative Decoding 反而会降低系统整体吞吐（Goodput）？
3. **KV-Outer 稀疏 Attention 边界**：在 Fireworks M3 Sparse Attention 中，为什么需要 Stage 1 的 `CountEdges` 对单个 KV 块设置 `replica_threshold = 128` 并做 Stage 2 的 LPT 负载均衡？如果所有 Query 块都集中命中最初的 4 个 System Prompt KV 块（Attention Sink）且不做副本拆分，B200 的 148 个 SM 会发生什么现象？

### 6.3 合成离线实验对照（`experiments/innovator_arch_tradeoffs/`）

为把上述四家创新公司的架构机制转化为可复算的定量边界，本仓库新增了原创标准库 CPU/offline 实验 [`experiments/innovator_arch_tradeoffs/`](../../experiments/innovator_arch_tradeoffs/README.md)（证据等级：`own-offline`，不声称硬件实测）：

1. **Fireworks M3 `KV-Outer` 字节账本精确复现（对齐 `docs/perf-results-b200.md` 与 `b300.md`）**：
   - 在 `Tq=4096, KV=32768, Hq=16, Hkv=1, topK=16, D=128, BF16, distinct_blocks=16 (M=4096)` 场景下，离线公式精确重算出 `fio1x = 0.295 GB`、`fioAct = 0.546 GB`、`cio = 0.294 GB`，与作者公布的 B200（`fwd_dev = 0.102 ms, 674 TFLOP/s, fbwAct = 5.36 TB/s`）及 B300（`fwd_dev = 0.091 ms, 758 TFLOP/s, fbwAct = 6.02 TB/s`）报告中的分析字节数完全一致；相比无 KV 复用基线（`ioAlt = 4.588 GB`）节省 **`8.40x`** 总前向 HBM I/O。
   - 完整的 ATen $\to$ CuTe AOT C-ABI $\to$ TMA Swizzle $\to$ SM100 `tcgen05` / `TMEM` 512 线程寄存器重分配（`176/176/96/64`）分派链路见 [张量 Layout/Dtype 到编译器与 SM100 Kernel 的分派链路](kernel-dispatch-layout.md)。
2. **Together Sequoia 投机树验证的 Roofline 拐点反例**：
   - 对 `70B FP16`（`140 GB` 权重，H100 级 `3.35 TB/s` HBM、`989 TFLOP/s` 峰值，拐点 `295.22 FLOP/B`），`K = 64` 投机树验证处于 Memory-bound 区间（算术强度 `64 FLOP/B`），期望接受 `4.2` tokens 时净吞吐提升 **`3.88x`**；但若盲目将树宽扩至 `K = 512`（算术强度 `512 FLOP/B > 295.22 FLOP/B`），验证阶段跨过 Roofline 拐点变为 Compute-bound（单步耗时从 `45.29 ms` 飙升至 `78.48 ms`），净加速比反而倒退至 **`2.94x`**。
3. **Groq LPU 与 Cerebras WSE-3 的单单元 SRAM 容量边界**：
   - 对 `70B INT8`（`70 GB` 权重，按 `75%` 权重可用率扣除激活/KV/指令区），单颗 Groq TSP/LPU（`220 MiB` SRAM）至少需 **`405` 颗芯片**串联才能实现纯片上 SRAM 驻留；单片 Cerebras WSE-3（`44 GB` 片上 SRAM，可用约 `33 GB`）同样超过单晶圆纯 SRAM 驻留上限（需 `>= 3` 片晶圆或依赖 MemoryX 外部流式权重加载）。

---

## 7. 源码深挖（2026-10-18 新增）：容量经济学、Prefix Cache Worker 路由、PCIe 双缓冲卸载与晶圆边缘 FIFO 反压

为与全栈第 10 层组件 [容量、成本与吞吐经济学](capacity-economics.md) 及实验 [`experiments/capacity_economics/`](../../experiments/capacity_economics/README.md) 形成闭环，本节进一步核验四家创新公司在**容量定容、缓存亲和路由、PCIe 带宽墙摊销与主机边缘 I/O 测量**上的五份一手源码：

### 7.1 Fireworks AI (`fw-ai/benchmark/llm_bench/prefill_load_test.py`)：多 Worker 路由头、`cached_tokens + 1` 预热与动态批大小

在固定提交 `fw-ai/benchmark@94cd111967c7be51a35777c475d7c5d2b66ec94a`（`2026-10-08`）的 [`llm_bench/prefill_load_test.py`](https://github.com/fw-ai/benchmark/blob/94cd111967c7be51a35777c475d7c5d2b66ec94a/llm_bench/prefill_load_test.py)（`blob: 2d1550e3d36455f95274502e85c03c44da7de138`，全文 `L1–648`）中，Fireworks 公开了用于评估多副本 Prefill 集群与 Prefix Cache 吞吐的完整基准测试器：

1. **二维 Worker 显式寻址头（`RoutingConfig`, `L42–60`）**：
   在包含 `num_servers` 台服务器、每台服务器运行 `num_gens` 个 Generator Worker 的分布式推理池中，若负载均衡器随机分发请求，同一个会话的 Prompt Cache 将在多台 Worker 间反复失效。`RoutingConfig.headers_for(flat)` 将扁平索引 `flat` 映射为两个显式 HTTP 路由头：
   - `x-fireworks-generator-worker-service-index = str(flat % num_servers)`
   - `x-fireworks-generator-worker-local-index = str(flat // num_servers)`
   从而强制将特定前缀预热和压测流量绑定到确定性的 `(srv, gen)` 实例上。
2. **为什么预热请求必须发送 `cached_tokens + 1` 个 Token？（`build_warmup_ids`, `L128–140`）**：
   源码在 `L129–130` 注释并实现了一个关键的推理引擎边界：`# Server caches prompt[:-1] (last token is first decode slot), so send cached_tokens + 1 tokens to get exactly cached_tokens cached`。因为在自回归生成中，输入序列的最后一个 Token 需要作为第一步生成（First Decode Slot）的 Query 输入参与 logits 计算，服务端 KV Cache 管理器仅将前缀 `prompt[:-1]` 固化为可复用缓存块。若预热时只发 `cached_tokens` 个 Token，实际写入缓存的只有 `cached_tokens - 1` 个 Token，后续请求会在第 `cached_tokens` 位置发生缓存部分未命中。
3. **按未缓存 Token 数反比缩放的动态 Batch 与最慢 Worker 木桶效应（`L297–318, L434–441`）**：
   - 每个请求实际需要 Prefill 矩阵乘计算的 Token 数为 `uncached_tokens = prompt_tokens - cached_tokens`，压测器据此动态计算单轮并发批大小：
     $$\text{batch\_size} = \max\!\left(1,\; \left\lfloor \frac{\text{min\_tokens\_to\_batch}}{\text{prompt\_tokens} - \text{cached\_tokens}} \right\rfloor\right)$$
   - 在多 Worker 并发轮次（`run_round`）中，所有 Worker 通过 `asyncio.gather` 同时压测，但整轮墙钟耗时由最慢的 Straggler Worker 决定：`duration = max(server_durations)`（`L306`），而服务端前向传播耗时取各 Worker 求和 `fwp = sum(fwps)`。这证明在多副本推理集群中，**Prefix Cache 命中率的不均匀分布会直接通过最慢 Worker 拖垮整批并发吞吐**。

### 7.2 Together AI / Sequoia (`Infini-AI-Lab/Sequoia/Engine/offload_engine.py`)：锁页内存双缓冲 PCIe 卸载与投机树摊销

当模型权重超出单机 GPU HBM 容量、不得不把部分 Transformer 层卸载到主机 CPU 内存时，传统逐 Token 自回归解码会遭遇灾难性的 PCIe 带宽墙。在固定提交 `Infini-AI-Lab/Sequoia@0ff3ff71475ca148f2dfb7545e0de5caf5c67b39`（`2025-01-28`）的 [`Engine/offload_engine.py`](https://github.com/Infini-AI-Lab/Sequoia/blob/0ff3ff71475ca148f2dfb7545e0de5caf5c67b39/Engine/offload_engine.py)（`blob: 7860ab7308cb94eb78f4e8514147257bdb3fa231`，全文 `L1–393`）中，`OffloadEngine` 展示了如何用“异步双缓冲 + 投机树验证”联合破解这一瓶颈：

1. **常驻层 `stay_layers` + 锁页内存 Ping-Pong 双缓冲（`L66–127, L179–215`）**：
   - `LlamaLayerBuffer` 为单层权重（`wq, wk, wv, wo, w1, w2, w3` 及 RMSNorm 权重）在 GPU HBM 上预分配固定显存槽位，并在初始化时将卸载到 CPU 的层通过 `.detach().pin_memory()` 锁定在主机页锁定内存（Pinned Memory）中以启用异步 DMA；
   - 引擎将前 `stay_layers` 层常驻 GPU（`self.layers[:stay_layers]`），仅为剩余 $L_{\text{off}} = L_{\text{total}} - L_{\text{stay}}$ 层在 GPU 上分配 **2 个 `LlamaLayerBuffer`（`self.buffer = [LlamaLayerBuffer(self.device) for _ in range(2)]`）** 与独立的 CUDA 流 `self.load_stream = torch.cuda.Stream(device=device)`；
   - 在前向循环 `inference()`（`L192–215`）中，当默认流正在计算第 `idx` 层（使用 `self.buffer[idx % 2]`）时，`self.load_stream` 并发执行 `self.buffer[(idx + 1) % 2].sync_copy(self.layers[idx + 1])`（调用 `.copy_(..., non_blocking=True)`），实现跨层 PCIe H2D 传输与 GPU 计算的 Ping-Pong 重叠。
2. **为什么投机树（Speculative Tree）对 PCIe Offload 具有乘数级成本收益？**：
   设单层字节数为 $W_{\text{layer}}$、PCIe 有效带宽为 $B_{\text{PCIe}}$，则每步必须从 CPU 搬运 $L_{\text{off}} \cdot W_{\text{layer}}$ 字节，单步耗时下界为：
   $$T_{\text{step, offload}} = \max\!\left(T_{\text{compute}},\; \frac{L_{\text{off}} \cdot W_{\text{layer}}}{B_{\text{PCIe}}}\right)$$
   由于 $B_{\text{PCIe}}$（PCIe Gen4/Gen5 约 `24–56 GB/s`）比 HBM3（`3,350 GB/s`）低 **$60\times\text{–}140\times$**，卸载模式下 $T_{\text{step, offload}}$ 几乎完全由 PCIe 传输时间支配，且验证 $K=64$ 个投机树候选 Token 与计算 1 个 Token 所需的 PCIe 权重传输量**完全相同**！因此在 [`experiments/capacity_economics/`](../../experiments/capacity_economics/README.md) 中，当 Sequoia 树每次验证平均接受 $\mathbb{E}[\tau] = 5.4\text{–}6.8$ 个 Token 时，在不增加任何一字节 PCIe 传输的前提下，有效生成吞吐从 `0.55–3.11 tok/s` 直接跃升至 **`2.96–21.16 tok/s`（净加速 `5.4x–6.8x`）**。

### 7.3 Groq (`groq/groqflow/groqflow/groqmodel/groqmodel.py`)：900 MHz 编译器周期、多芯片流水乘子与 PCIe 固定开销方程

在固定提交 `groq/groqflow@720e024414be401a7853931a2622a26446883746`（`2023-12-19`）的 [`groqflow/groqmodel/groqmodel.py`](https://github.com/groq/groqflow/blob/720e024414be401a7853931a2622a26446883746/groqflow/groqmodel/groqmodel.py)（`blob: 9e07c49316833654cc8e1ca5309e8247837b198a`，全文 `L1–460`）中，`GroqModel.estimate_performance()`（`L116–164`）给出了 Groq 官方工具链在无需物理上板时预测端到端性能的精确解析模型：

```python
# groqflow/groqmodel/groqmodel.py:130-156
frequency = 900000000  # Hz (900 MHz)
pcie_latency = 1e-5  # seconds (10 us)
pcie_bandwidth = 24e9  # bytes per second (24 GB/s)

self.state.info.estimated_pcie_input_latency = (
    self.state.info.compiled_model_input_bytes / pcie_bandwidth
) + pcie_latency
self.state.info.deterministic_compute_latency = on_chip_compute_cycles / frequency

# When pipelined, the reported cycle is the duration of a single pipelining stage
# Note: Models are pipelined by default
if not "--no-multichip-pipelining" in self.state.config.compiler_flags:
    self.state.info.deterministic_compute_latency *= self.state.num_chips_used

self.state.info.estimated_pcie_output_latency = (
    self.state.info.compiled_model_output_bytes / pcie_bandwidth
) + pcie_latency
```

这段源码从工程上证实了三个关键事实：
1. **单级周期与多卡端到端延迟的乘数关系**：当模型跨 $N_{\text{chips}}$ 张 LPU 编译且默认启用多芯片流水线（未传 `--no-multichip-pipelining`）时，编译器 `stats_file` 中的 `total_cycles` 仅代表**单个流水线级（单张芯片）的周期数**。因此，单次推理走完 $N_{\text{chips}}$ 张芯片的确定性片上计算延迟必须乘以 `self.state.num_chips_used`（与 4.2 节 `execute.py` 中连续调用 $N_{\text{chips}}$ 次 `tsp_runner` 严格对应）！
2. **吞吐与延迟的分离**：在持续满载流水线模式下，系统每秒可完成的样本吞吐取决于单级周期 $f / \text{total\_cycles}$，但单个 Token Step 的端到端延迟取决于 $N_{\text{chips}} \cdot (\text{total\_cycles} / f)$。
3. **PCIe 往返延迟下界**：每次与主机交互固定叠加输入与输出两次 `10 µs`（共 `20 µs`）PCIe 事务延迟及 `24 GB/s` 带宽项。

### 7.4 Cerebras (`Cerebras/sdk-examples/benchmarks/bandwidth-test/`)：100GbE 16 通道边缘 I/O、46 KB/PE FIFO 反压缓冲列与 Manhattan 时钟补偿

在固定提交 `Cerebras/sdk-examples@d19f2ca3e51285fc5904c78643b38819a2f26c44`（`2026-04-27`）的 [`benchmarks/bandwidth-test/run.py`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/bandwidth-test/run.py)（`blob: fd1d87d73f1cb650e68688507be58e62d92e3790`，全文 `L1–402`）与 [`src/bw_sync_kernel.csl`](https://github.com/Cerebras/sdk-examples/blob/d19f2ca3e51285fc5904c78643b38819a2f26c44/benchmarks/bandwidth-test/src/bw_sync_kernel.csl)（`blob: 3060e774cc14e05d523a1d5d7e11140aafc0c376`，全文 `L1–145`）中，源码注释与实现完整披露了晶圆级芯片与外部主机通信的物理瓶颈与面积开销：

1. **100 Gbps 以太网边缘链路与 16 通道切分（`run.py:16–64`）**：
   主机与 WSE 之间通过 `100 Gbps` 以太网连接，数据被切分到最多 `16` 个并行 I/O 通道（`--channels`，单通道峰值约 `7 Gbps` $\approx 875\text{ MB/s}$），且每次主机侧 `memcpy_h2d` / `memcpy_d2h` 调用都带有约 **`200 µs` 的 TCP 事务开销**。
2. **反压（Backpressure）与 `46 KB/PE` 硬件 FIFO 缓冲列开销（`run.py:34–64, 92–122`）**：
   - 若目标计算 PE 正在执行繁重计算而无法及时响应 I/O Wavelet，反压会沿片上织物一路回传并阻塞主机 I/O 通道；
   - 为解耦主机传输与片上计算，编译器在计算网格西侧（H2D 入口）与东侧（D2H 出口）插入专门的缓冲列 `--width-west-buf` 与 `--width-east-buf`，其中每个缓冲 PE 将自身几乎全部的 **`46 KB` SRAM 配置为硬件 FIFO**：
     $$C_{\text{buf}} = \left\lceil \frac{W \cdot L_{\text{pe}} \cdot 4\text{ bytes}}{46{,}000\text{ bytes}} \right\rceil$$
   - 同时，`memcpy` 运行时框架自身还必须在西侧占用 `3` 列系统 PE、在东侧占用 `2 + 1 = 3` 列系统 PE（`run.py:92–95`）：
     $$\texttt{core\_fabric\_offset\_x} = x_{\text{offset}} + 3 + C_{\text{west}}, \qquad W_{\min} = \texttt{core\_fabric\_offset\_x} + W + 2 + 1 + C_{\text{east}}$$
     例如在 [`experiments/capacity_economics/`](../../experiments/capacity_economics/README.md) 的场景 2（$W = 256, L_{\text{pe}} = 512$，单次传输 `524,288 bytes`）中，仅为了无阻塞缓冲一次 H2D/D2H 传输，西侧与东侧就各需 **12 列 FIFO 缓冲 PE + 3 列系统 PE**，使最小织物宽度从 `256` 膨胀至 **`287` 列（额外消耗 `12.1%` 的晶圆列宽）**！
3. **48-Bit 时间戳 `f32` 位重解释与 2D Manhattan 波前延迟补偿（`bw_sync_kernel.csl:74–144` & `run.py:216–244, 336–378`）**：
   - 在 `bw_sync_kernel.csl` 中，每个 PE 通过 `timestamp.get_timestamp(&tscStartBuffer)` 采样 3 个 `u16` 字（48-bit 周期计数器），并通过 `bitcast` 将相邻 `u16` 拼接打包进 `f32` 数组传回主机；
   - 由于 `sync/pe.csl` 的全晶圆同步广播波前从左上角 `PE(0,0)` 沿东北$\to$东南方向逐跳扩散，坐标为 `(px, py)` 的 PE 收到广播信号并锁存 `time_ref` 的时刻比 `PE(0,0)` 晚了恰巧 `px + py` 个周期。因此 `run.py:240–244` 显式执行 **Manhattan 波前补偿**：
     $$T_{\text{ref, comp}}(p_y, p_x) = T_{\text{ref}}(p_y, p_x) - (p_x + p_y)$$
     再计算全网格最早开始周期 $\min(T_{\text{start}} - T_{\text{ref, comp}})$ 与最晚结束周期 $\max(T_{\text{end}} - T_{\text{ref, comp}})$ 之差 `cycles_send`，按 `0.85 GHz` 换算微秒并得出真实片上边缘带宽。

练习状态：`not-assessed`。后续可证伪研究问题已登记至 [`research/queue.json`](../../research/queue.json)。

