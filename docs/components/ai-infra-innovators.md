# AI Infra 创新公司深挖：推理加速平台（Fireworks / Together）与新型 AI 芯片（Groq / Cerebras）

状态：2026-10-14，基于固定开源代码提交与同行评审/预印本论文正文的**有界公开研究**。
先修：[GPU/HBM 深入研究](gpu-hbm.md)、[互连与网络分层](interconnect-network.md)。
**严格边界声明**：本文仅依据公开源码（`implementation`）、架构论文（`specification` / `author-benchmark`）及本仓库合成公式推导立论；**未调用任何商业付费 API，未在 B200/H100/Groq LPU/Cerebras WSE 硬件上实测**，作者披露的跑分均显式标注为 `author-benchmark`，绝不冒充本项目独立重测（`own-hardware`）。

证据 ID：`fireworks-m3-sparse-sm100`、`fireworks-b200-author-bench`、`fireworks-locust-cache-bench`、
`together-fa3-hopper-async`、`together-sequoia-spec-tree`、`together-stripedhyena-state`、`together-moa-fanout`、
`groq-tsp-microarch-bert`、`groq-groqflow-multichip`、`cerebras-csl-wse-allreduce`、`cerebras-wse-io-roofline`。
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

在 [`fw-ai/benchmark@94cd111967c7be51a35777c475d7c5d2b66ec94a`](https://github.com/fw-ai/benchmark/blob/94cd111967c7be51a35777c475d7c5d2b66ec94a/llm_bench/load_test.py)（`2026-04-16`，`implementation`）中，`llm_bench/load_test.py` 展示了推理平台压测最容易失真的变量控制：
- 若每次请求使用完全相同的 prompt，服务端 Prefix Cache（RadixAttention / Prompt Cache）会 100% 命中，测出的 TTFT 只是缓存读取延迟而非真实 Prefill 算力；
- 该代码通过 `--prompt-cache-max-len` 显式构造 `common_tokens`（共享前缀长度）与每次随机采样的后缀 token 拼接（`TranslationDataset.sample`），并分别上报 `time_to_first_token`（TTFT）、`latency_per_token`（TPOT）、`generation_tokens_per_second` 与 `full_tokens_per_second`。这验证了评估任何商用推理 API 时，**必须显式锁定共享前缀比例（Prefix Cache Hit Ratio）与输出长度分布**。

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
   - **架构含义**：Cerebras WSE 只有在**权重常驻晶圆片上 SRAM、避免频繁从主机流式换入权重**时才能发挥晶圆级带宽优势；一旦模型权重超出单晶圆总 SRAM 容量需要频繁跨晶圆/主机流式加载，系统吞吐会立即退化到边缘 I/O 通道上限。

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

练习状态：`not-assessed`。后续可证伪研究问题已登记至 [`research/queue.json`](../../research/queue.json)。
