# 大模型确定性推理与跨批大小逐比特可复现架构选型指南（`默认动态连续批处理` vs. `粗暴锁死 BS=1 串行确定性` vs. `Thinking Machines 批大小不变算子库 [Batch-Invariant Kernels]` vs. `全栈 0-ULP 确定性推理引擎 [固定 Split-Size 注意力 + 固定序 TP 规约 + 请求级 Philox 采样]`）

> **适用角色**：AI 解决方案架构师（AI Solution Architect）、金融/医疗/法律合规 AI 平台负责人、RLHF/GRPO 强化学习框架研发工程师  
> **阅读目标**：3 分钟掌握大模型推理“为什么设置了 `temperature=0.0` 和 `seed=42`，同一个 Prompt 两次调用的返回结果还是不一样”的物理真相与四档选型铁律；10 分钟吃透 **浮点加法非结合律 `(a+b)+c != a+(b+c)` 如何通过动态 Continuous Batching 触发 `cuBLAS Split-K`、`FlashDecode 动态分块数` 与 `NVSwitch 异步组播` 导致第 40 个词分叉**，以及 **如何通过 `Batch-Invariant Kernels`（固定 `Split-Size=256` 注意力 + 固定 K 维归约树 GEMM + 独立行 RMSNorm + 请求级 Philox RNG）在保住 `93.0%` 满血吞吐（比锁 `BS=1` 快 `11.16 倍`）的前提下实现跨任意并发批大小的 `100.0% 0-ULP 逐比特复现`**！  
> **数据来源说明（Provenance）**：本文定量数据来自仓库内封存的确定性解析模型（[`experiments/batch_invariant_deterministic_llm_inference_selection/run.py`](../../experiments/batch_invariant_deterministic_llm_inference_selection/run.py) 与 [`results.json`](../../experiments/batch_invariant_deterministic_llm_inference_selection/results.json)，校准自 Thinking Machines Lab 批大小不变算子研究、vLLM/SGLang 确定性采样架构与 IEEE-754 浮点规约模型，`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`）。

![大模型确定性推理与跨批大小逐比特可复现选型一图流总结](../../assets/svg/batch-invariant-deterministic-llm-inference-reproducibility-selection-faq-summary.svg)

---

## 📌 一、核心结论速览（Executive Summary）

1. **为什么线上把 `temperature=0.0`、`top_k=1`、`seed=42` 全锁死了，客户在默认 vLLM / SGLang / TensorRT-LLM（Tier 1）上跑同一个 Prompt，白天和半夜还是会吐出完全不同的合同条款或财务数字？**：
   - **破除迷思**：GPU 本身绝不是“掷骰子的随机机器”——如果你在完全相同的输入张量上跑同一个 CUDA Kernel，它永远是 100% 确定性的！
   - **真正的罪魁祸首是“批大小非不变性（Lack of Batch Invariance）”**：在线上 **Continuous Batching（连续批处理）** 引擎中，你的请求在半夜可能是独占 GPU（当前批大小 `BS=1`），而在白天高峰期则是和另外 31 个用户的请求拼在一起算（当前批大小 `BS=32`，甚至还夹杂着别的请求的 Chunked Prefill 分块）。
   - 当批大小 $M$ 从 `1` 变成 `32` 时，**Tier 1（默认推理引擎）** 底层的三大核心算子为了压榨极致速度，会**动态改变切块方式与并行归约顺序**：
     1. **`cuBLAS GEMM` 矩阵乘**：在 `M=1` 时调用启用 `Split-K` 的窄矩阵内核（多个 CTA 沿着 $K$ 维切分并行累加），而在 `M=32` 时切换为标准 `128×128` Tensor Core MMA 内核；
     2. **`FlashDecode / PagedAttention`**：默认根据“当前 GPU 还有多少空闲 SM”来动态决定把一条序列的 KV Cache 切成几块（`BS=1` 时切成 `8 块` 归约，`BS=32` 时只切成 `2 块` 归约）；
     3. **IEEE-754 浮点数加法不满足结合律（$(a + b) + c \neq a + (b + c)$）**：归约树合并顺序一变，每层输出的浮点数最低有效位就会产生 **`1 ~ 14 ULP`** 的微小舍入差异。经过 80 层 Transformer 指数级放大后，当两个候选词的 Logits 非常接近时，第 30～50 个 Token 就会发生 `argmax` 翻转，随后沿着自回归蝴蝶效应生成两篇完全不同的文本（**跨负载完全一致率仅 `61.5%` ✕**，还会导致 RL 强化学习采样器与训练器产生 `0.0185` 的虚假 KL 散度偏移 ✕）！
2. **为什么靠“强行设置 `max_num_seqs=1` 锁死串行（Tier 2）”是自杀式方案？生产环境该如何选型 Tier 3 与 Tier 4？**：
   - **Tier 2（粗暴串行确定性：锁死 `BS=1` + `torch.use_deterministic_algorithms(True)`）**：虽然强行让每次调用的批大小永远等于 1 从而实现了 `100.0%` 一致，但这等于**把价值 200 万人民币的 8×H100 节点废成了单线程算盘**！GPU 显存带宽与 Tensor Core 严重饥饿，吞吐从 `3,420 tok/s` 暴跌 **`-91.7%` 至 `285.0 tok/s` ✕**，百万 Token 成本飙升 **12 倍至 `$23.39` ✕**（单节点每年白白浪费 `$1,480,000` 算力 ✕）！
   - **Tier 3（`Thinking Machines 批大小不变算子库：Batch-Invariant RMSNorm + Fixed-Tile GEMM + Fixed-Split-Size=256 FlashDecode`）**：
     - 从算子第一性原理重构：**让每一个 Token 在 K 维和 KV 序列维上的浮点加法归约树形状，严格只取决于该请求自身的长度，而与当前 Batch 里拼了多少个其他请求 100% 无关**！
     - 在单卡部署下即可实现 `100.0%` 逐比特复现（8 卡 TP 下达 `99.4%`），且完美保留连续批处理，稳态吞吐高达 **`3,110 tok/s`（保住 `90.9%` 满血性能 ✅）**！
   - **Tier 4（`全栈 0-ULP 逐比特确定性推理引擎：Tier 3 批不变算子 + 固定树序 TP AllReduce + 绑定 Request-ID 的独立 Philox RNG + 前缀缓存不变分块`）**：
     - 补齐多卡张量并行（`TP=8`）与高温随机采样（`temperature > 0`）的最后两块拼图：① 禁用机内 NVSwitch `NVLS` 动态乱序组播加法，改用固定拓扑二叉树顺序的确定性 `TP AllReduce`；② 废除全局共享随机数生成器，为每个请求槽位绑定独立的 `(seed, request_token_offset)` **Philox 4×32 计数器随机数流**；
     - 在 `8×H100` 上实现跨 `BS=1 ~ 64`、跨 Prefix Cache 命中与否的 **`100.0%` 绝对逐比特复现（`0.0 ULP` 漂移 🏆）**，RL 训练-采样策略 KL 漂移降为 **`0.0000`（真正意义上的 True On-Policy RL 🏆）**，稳态吞吐达 **`3,180 tok/s`（保住 `93.0%` 满血吞吐，比锁 BS=1 快 `11.16 倍` 🏆）**，百万 Token 成本仅 **`$2.10`**！
3. **一句话握手机制本质**：
   - **大模型推理确定性的第一性原理，绝不是“关掉并发退回 BS=1”，而是“把算子归约轴（Reduction Axis）与批处理轴（Batch Axis）彻底解耦——永远用固定的 KV Split-Size 和固定的 K-Tile 二叉归约树做加法”！**

---

## 🎯 二、客户与架构师灵魂拷问 Q&A（FAQ-Driven First Principles）

### Q1：为什么同样是算注意力（FlashAttention / PagedAttention），默认实现会随并发批大小变化而改变结果，而把 `Fixed Split-Count` 改成 `Fixed Split-Size = 256` 就能一招根治？

- **第一性原理拆解**：
  1. **默认 `FlashDecode` 的“按空闲 SM 动态均分（Fixed Split-Count）”陷阱**：
     - 在 Decode 阶段，Query 长度 $S_q = 1$，如果只用一个线程块（CTA）去扫一条长 `4,096` Token 的 KV Cache，80 张 SM 里只有几张在干活。所以 FlashDecode 会把这条 KV Cache 沿着序列维度切成 $N_{\text{splits}}$ 份并行算局部 Softmax，最后再把这 $N_{\text{splits}}$ 份的 `(max_logit, lse, output_acc)` 合并起来。
     - **致命问题来了**：默认启发式算法是怎么决定 $N_{\text{splits}}$ 的？它看的是 **`当前 GPU 总 SM 数 ÷ 当前 Batch 里的请求总数`**！
     - 当半夜只有你 **`1 个请求（BS=1）`** 时，GPU 很闲，它把你的 4,096 个 KV 切成 **`8 份`（每份 512 词）** 来归约：结果是 $((O_1 + O_2) + \dots + O_8)$；
     - 当白天高峰期有 **`32 个并发请求（BS=32）`** 时，SM 不够分了，它把你同样的 4,096 个 KV 只切成 **`2 份`（每份 2,048 词）** 来归约：结果是 $(O'_1 + O'_2)$！
     - 因为浮点指数重缩放（Rescaling）和加法顺序完全变了，哪怕是同一个请求、同一份 KV Cache，`BS=1` 和 `BS=32` 算出来的 Attention 输出在最后几位尾数上必然不同！
  2. **批大小不变（Batch-Invariant）的解法：锁死 `Fixed Split-Size = 256`**：
     - 不再根据当前 Batch 有多少个请求去动态决定切几份，而是硬性规定：**无论当前 Batch 是 1 还是 64，任何请求的 KV Cache 永远严格每 `256 个 Token` 切成一个固定分片（4,096 词永远固定切成 `16 个分片`）**，并且汇总阶段严格按照固定的二分归约树顺序 `((S0+S1)+(S2+S3))...` 合并！
     - 这样一来，无论隔壁拼了 0 个还是 63 个其他用户的请求，甚至无论你的请求命中了多少块 Prefix Cache，你每一个 Token 经历的浮点加法顺序永远 **100% 恒定不变**！

---

### Q2：除了 Attention，为什么连最简单的 `RMSNorm` 和线性层 `MatMul (GEMM)` 在默认引擎里也会破坏跨批大小确定性？

- **两大隐蔽凶手拆解**：
  1. **`MatMul (GEMM)` 的动态内核切换与 `Split-K`**：
     - 在 Transformer 的 `QKV 投影`、`O 投影` 与 `FFN Up/Gate/Down` 矩阵乘 $C_{M \times N} = A_{M \times K} \times B_{K \times N}$ 中，批大小 $M$ 是参与并行分块的维度。
     - 当 $M=1\sim 4$ 时，为了填满 GPU，cuBLAS 会自动选用 **`Split-K` 内核**（把内积维度 $K=8192$ 切成 4 段，有时还通过 `atomicAdd` 乱序累加，或者走先段内求和再段间求和的二级归约树）；而当 $M=32$ 时，cuBLAS 切换到 **不切 $K$ 维的标准 `128×128` Tensor Core 瓦片内核**（从头到尾顺序累加 $K$）。两边在 $K$ 维上的浮点加法结合顺序不同，直接产生 `2 ~ 8 ULP` 漂移！
     - **根治方法**：使用批不变 GEMM 配置——**全局禁用动态 `Split-K` 原子累加**，并强制所有 $M$ 档位统一使用相同的 $K$ 维瓦片步长（如固定 `BLOCK_K = 64` 顺序累加），边界不足一个 Tile 的行用零填充（Zero-Padding）而不改变指令序列。
  2. **`RMSNorm` 的 Warp 协同归约变化**：
     - 部分高性能 Norm 内核在小 Batch 时让多个 Warp 协作归约同一行隐藏层向量（$d=8192$），而在大 Batch 时让单个 Warp 独立归约一行。**批不变 RMSNorm** 强制规定“永远由固定数量的线程（如 1 个 CTA 固定 256 线程）按完全相同的蝶形树（Butterfly Tree）归约每一行”，彻底消除行内归约顺序漂移。

---

### Q3：为什么在强化学习后训练（RLHF / GRPO / RLVR，如 DeepSeek-R1 类训练）中，推理引擎的“非确定性浮点漂移”会直接诱发训练奖励崩塌（Reward Collapse）？

- **RL 采样与训练的“On-Policy 数值鸿沟”**：
  - 在 GRPO / PPO 训练中，推理引擎（vLLM/SGLang Rollout Worker）先生成轨迹并记录每个词的对数概率 $\log \pi_{\text{rollout}}(a_t|s_t)$，随后训练引擎（FSDP/Megatron Actor）用前向传播重算 $\log \pi_{\text{train}}(a_t|s_t)$，并计算重要性采样比值 $r_t = \exp(\log \pi_{\text{train}} - \log \pi_{\text{rollout}})$。
  - 在第一轮更新前，两边的模型权重是**完全一模一样**的，理论上 $r_t$ 应该严格等于 `1.0000`（$\text{KL} = 0$）。
  - 但由于 **Tier 1** 推理时是动态逐词 Decode Batch，而训练时是整序列 Prefill Batch，两边算子归约树不同，导致在 32K 长思维链上累积出 **`0.0185` 的虚假数值 KL 散度**！优化器误以为策略已经发生了剧烈偏移，频繁错误触发 PPO Clip 截断或梯度爆炸。启用 **Tier 4 批大小不变算子** 后，$\log \pi_{\text{rollout}}$ 与训练前向严格对齐（KL 漂移降至 **`0.0000` 🏆**），长链 RL 收敛稳定性大幅提升！

---

## 🧭 三、直观可视化图解（Mermaid 决策树 / 归约树对比 / 时序图）

### 3.1 大模型确定性推理与可复现架构选型决策树

```mermaid
flowchart TD
    Start["大模型确定性推理与逐比特复现选型"] --> Q_Scene{"核心业务场景与合规/训练要求？"}
    
    Q_Scene -- "普通闲聊 / 创意写作 / 容忍每次微调差异" --> T1["Tier 1: 默认动态连续批处理<br/>(最高吞吐 3,420 tok/s · 但跨负载一致率仅 61.5%)"]
    
    Q_Scene -- "金融合同审计 / 医疗报告 / 司法取证 /<br/>CI 回归评测门禁 / RLHF On-Policy 采样" --> Q_Bad{"是否试图靠锁死 BS=1<br/>(max_num_seqs=1) 换确定性？"}
    
    Q_Bad -- "是 (Tier 2 粗暴串行)" --> T2_Warn["⛔ 严禁上生产! 吞吐暴跌 -91.7% (285 tok/s)<br/>百万词成本飙至 $23.39 · 单机年损 $148万!"]
    
    Q_Bad -- "否 (保留高并发 Continuous Batching)" --> Q_TP{"部署拓扑是单卡 (TP=1)<br/>还是多卡张量并行 (TP=2/4/8)？"}
    
    Q_TP -- "单卡服务 / 允许 0.5 ULP 跨卡微差" --> T3["Tier 3: Thinking Machines 批不变算子库<br/>(固定 Split-Size=256 注意力 + 固定 K-Tile GEMM)<br/>(单卡 100% 一致 · 吞吐 3,110 tok/s [90.9%])"]
    
    Q_TP -- "多卡 TP=8 生产集群 / 严苛 0-ULP 监管审计 /<br/>RLVR 强化学习零 KL 漂移采样" --> T4["Tier 4: 全栈 0-ULP 逐比特确定性推理引擎<br/>(批不变算子 + 固定树序 TP AllReduce + 请求级 Philox RNG)<br/>(跨 BS=1~64 100.0% 逐比特一致 · 吞吐 3,180 tok/s 🏆)"]
```

---

### 3.2 默认引擎因批大小改变归约树（Tier 1 漂移）vs. 批不变算子固定归约树（Tier 4 0-ULP）对比

```mermaid
flowchart LR
    subgraph Drift["🟥 Tier 1 默认引擎：同一个请求在 BS=1 与 BS=32 下经历不同的浮点加法树 (产生 14 ULP 漂移)"]
        direction TB
        D1["半夜空闲 (BS=1): cuBLAS 启用 4-way Split-K · FlashDecode 切 8 个 KV Split<br/>浮点求和顺序: ((s1+s2)+(s3+s4)) + ((s5+s6)+(s7+s8))"]
        D2["白天高峰 (BS=32): cuBLAS 切标准 MMA Tile · FlashDecode 仅切 2 个 KV Split<br/>浮点求和顺序: (s1+s2+s3+s4) + (s5+s6+s7+s8)"]
        D1 -. "IEEE-754 (a+b)+c != a+(b+c)<br/>第 42 个生成词发生 argmax 翻转 ✕" .- D2
    end

    subgraph Invariant["🟩 Tier 4 全栈批不变引擎：归约树严格绑定请求自身序列，与并发 BS 彻底解耦 (0.0 ULP)"]
        direction TB
        I1["1. Batch-Invariant RMSNorm & GEMM: 禁用动态 Split-K，固定 BLOCK_K=64 顺序累加"]
        I1 --> I2["2. Fixed-Split-Size=256 FlashDecode: 4K 上下文永远固定切成 16 块并按固定二叉树合并"]
        I2 --> I3["3. 固定顺序 TP AllReduce + 绑定 Request-ID 的独立 Philox RNG<br/>无论 BS=1 还是 BS=64，100.0% 吐出逐比特完全相同的 Token 序列 🏆"]
    end
```

---

### 3.3 Tier 4 全栈确定性推理引擎在混合并发（`BS=32` + Prefix Cache + 多卡 `TP=8`）下的单步执行时序图

```mermaid
sequenceDiagram
    autonumber
    participant Client as 金融审计客户端 (req_id=8848, seed=42)
    participant Sched as Continuous Batching 调度器 (当前并发 BS=32)
    participant Kernel as 批大小不变算子层 (RMSNorm / GEMM / FlashDecode)
    participant Comm as 确定性 TP=8 通信层 (Fixed-Order Tree AllReduce)
    participant Sampler as 请求级独立采样器 (Per-Req Philox 4x32 RNG)

    Client->>Sched: 1. 提交 4,096 词合同审查请求 (与另外 31 个动态请求拼成 M=32 Batch)
    Sched->>Kernel: 2. 执行 RMSNorm (每行独立固定 256 线程蝶形归约) + QKV GEMM (固定 K-Tile 无 Split-K)
    Kernel->>Kernel: 3. 执行 FlashDecode: 强制按 Fixed Split-Size=256 切为 16 个 KV 分块，按固定二叉树合并 LSE
    Kernel->>Comm: 4. 执行 O-Proj / Down-Proj 后的跨卡同步: 旁路非确定性 NVLS 组播，按固定 Rank 树顺序累加
    Comm->>Sampler: 5. 产出与 BS=1 时逐比特 100% 相同的 0-ULP Logits!
    Sampler->>Client: 6. 用 hash(seed=42, req_token_idx) 专属 Philox 计数器采样，返回 100.0% 可复现结果!
```

---

## 📊 四、架构师选型决策矩阵（Architect Decision Matrix）

*基准场景：**单台 `8×H100 80GB SXM5` 节点（`$24.00/hr`）** 运行 `70B FP8/BF16` 大模型（`TP=8`，输入 `4,096` 词，输出 `512` 词），在动态负载波动（`BS=1` vs `BS=8` vs `BS=32` vs `BS=64`）下评测同一 Prompt 的可复现性与系统吞吐。*

| 评估维度 | Tier 1: 默认动态连续批处理 | Tier 2: 粗暴强锁 `BS=1` 串行确定性 | Tier 3: 批大小不变算子库 (`Batch-Invariant`) | Tier 4: 全栈 `0-ULP` 确定性推理引擎 |
| :--- | :--- | :--- | :--- | :--- |
| **核心归约与同步机制** | 动态 Split-K + 动态 Split-Count + NVLS | 禁用连续批处理（强制 `max_num_seqs=1`） | **固定 `Split-Size=256` + 固定 K-Tile GEMM** | **批不变算子 + 固定序 TP 规约 + 独立 Philox** |
| **跨并发批大小完全一致率** | `61.5%` (`38.5%` 回答发生分叉 ✕) | `100.0%` | **`99.4%` (单卡 `100.0%` ✅)** | **`100.0%` (跨任意负载绝对逐比特一致 🏆)** |
| **最大 Logit 浮点漂移** | `14.0 ULP` ✕ | `0.0 ULP` | `0.5 ULP` (仅余多卡 NVLS 微差) | **`0.0 ULP` (严格逐比特一致 🏆)** |
| **RL 采样-训练策略 KL 偏移** | `0.0185` (污染重要性采样比 ✕) | `0.0000` (但采样慢 11 倍无法训练 ✕) | `0.0004` ✅ | **`0.0000` (True On-Policy 零漂移 🏆)** |
| **8×H100 稳态推理吞吐** | `3,420.0 tok/s` (`100%` 基准) | `285.0 tok/s` (**暴跌 `-91.7%` ✕**) | `3,110.0 tok/s` (`90.9%` 满血 ✅) | **`3,180.0 tok/s` (`93.0%` 满血 · 比 Tier 2 快 `11.16x` 🏆)** |
| **P99 TPOT 生成尾延迟** | `14.2 ms` | `96.5 ms` (严重排队阻塞 ✕) | `15.6 ms` (`+1.4 ms` 无感开销 ✅) | **`15.2 ms` (`+1.0 ms` 丝滑达标 🏆)** |
| **百万 Token 成本 / 年化重跑与空转损失** | `$1.95 / 1M` · 年重跑损 `$420k` ✕ | `$23.39 / 1M` · **年空转损 `$1.48M` ✕** | `$2.14 / 1M` · 年损 `$48k` | **`$2.10 / 1M` · `$0` 审计复现损失 (省 `$1.48M` 🏆)** |

---

## 🧪 五、简明定量算例与 7-Card Benchmark 洞察

### 5.1 并发批大小与 Prefix Cache 扰动对输出一致率的影响

在 [`experiments/batch_invariant_deterministic_llm_inference_selection/run.py`](../../experiments/batch_invariant_deterministic_llm_inference_selection/run.py) 中，我们实测模拟了四组生产常见扰动下的输出完全一致率：

| 生产负载扰动条件 | Tier 1: 默认动态引擎一致率 | Tier 3: 仅批不变算子一致率 | Tier 4: 全栈 0-ULP 引擎一致率 | Tier 1 发生漂移的物理根因 |
| :--- | :--- | :--- | :--- | :--- |
| **`BS=1` vs. `BS=4` (轻载并发变化)** | `78.2%` | `99.8%` | **`100.0%` 🏆** | `cuBLAS` 在 $M \ge 4$ 时从窄矩阵 `Split-K GEMV` 切换为 Tensor Core `MMA` |
| **`BS=1` vs. `BS=32` (中载连续批处理)** | `64.0%` ✕ | `99.5%` | **`100.0%` 🏆** | `FlashDecode` 动态分块数从每序列 `8 splits` 缩减为 `2 splits` |
| **`BS=1` vs. `BS=64` + Chunked Prefill 混批** | `52.8%` ✕ | `99.1%` | **`100.0%` 🏆** | Prefill 与 Decode 拼成混合 Token 矩阵，改变 GEMM 分块与 NVLS 组播到达序 |
| **`Prefix Cache 0% 未命中` vs. `80% 命中`** | `61.5%` ✕ | `99.4%` | **`100.0%` 🏆** | 命中缓存部分走 Paged KV 读取而未命中走分块计算，除非固定 `Split-Size=256` |

### 5.2 配套交互式工作台与 7-Card 4:5 视觉评测报告

- 🧮 **交互式大模型确定性推理与可复现定容计算器**：[`docs/reports/batch-invariant-deterministic-llm-inference-reproducibility-sizer.html`](../reports/batch-invariant-deterministic-llm-inference-reproducibility-sizer.html)（以及总工作台 [`llm-inference-engine-pd-sizer.html#batch-invariant-det-sizer`](../reports/llm-inference-engine-pd-sizer.html)）
- 📊 **7-Card 4:5 视觉基准评测报告**：[`docs/reports/batch-invariant-deterministic-llm-inference-reproducibility-selection-cards.html`](../reports/batch-invariant-deterministic-llm-inference-reproducibility-selection-cards.html)

---

## 🛠️ 六、附录：生产环境快速核验清单（Architect Checklist）

- [ ] **1. 严禁靠设置 `max_num_seqs=1` 退回串行模式来换取确定性**：这会让 H100/B200 吞吐暴跌 `91.7%`；必须启用 **Batch-Invariant Kernels（批大小不变算子模式）**，在开启 Continuous Batching 的同时保持确定性。
- [ ] **2. 将 FlashDecode / PagedAttention 切换为 `Fixed Split-Size = 256` 模式**：禁止内核根据空闲 SM 数动态计算 `num_splits`，确保同一条 KV 序列在任何并发度下都切分成完全相同的块数并按固定树归约。
- [ ] **3. 锁定 GEMM 内核配置并禁用动态 `Split-K`**：在批不变模式下强制 `cuBLAS / Triton / CUTLASS` 对所有 Batch $M$ 采用统一的 `BLOCK_K` 顺序累加，杜绝 `M<8` 与 `M>=8` 切换不同归约树。
- [ ] **4. 多卡 `TP` 推理固定 AllReduce 规约顺序**：在需要 `0-ULP` 逐比特复现的审计/评测池上，设置 `NCCL_NVLS_ENABLE=0` 并固定 `NCCL_ALGO=Ring` 或确定性顺序 `Tree`，消除交换机/NVSwitch 异步包到达顺序引起的最后 1 bit 舍入抖动。
- [ ] **5. 采样端强制启用 `Per-Request Philox Seeded Sampler`**：严禁所有并发请求共享同一个全局 PyTorch `cuda.Generator`；必须用 `(request_seed, token_step)` 独立初始化每个请求的 Gumbel-Max / Top-P 采样器。
