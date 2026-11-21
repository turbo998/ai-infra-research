# PD 分离（`Prefill-Decode Disaggregation`）深度原理解析与适用场景边界：为什么 Prefill 与 Decode 混部会相互阻塞、何时保持混部（`Colocated + Chunked Prefill`）vs. 何时必须上 PD 分离、KV Cache 跨机传输带宽手算与主流框架（`Dynamo / Mooncake / SGLang / vLLM`）选型指南

> **适用读者**：正在设计高并发大模型在线推理平台、需要在“单池混部（Colocated Continuous Batching + Chunked Prefill）”与“预填充-解码分离（PD Disaggregation）”之间做精准架构拍板的 Cloud & AI Infrastructure Solution Architect、推理引擎技术负责人与智算网络架构师。
> **核心课题**：大模型推理的 **Prefill（首字预填充）** 与 **Decode（逐字自回归解码）** 为什么天生是一对“水火不容”的计算阶段？当一个 `16,384` Tokens 的长上下文 RAG/Agent 请求进入正在流式吐字的混部 GPU 时，为什么会把同卡所有用户的吐字间隔（`P99 TPOT`）从 `16ms` 瞬間拉爆到 **`148ms`（即使用上 `Chunked Prefill` 依然高达 `58.5ms`，突破 `<=25ms` 实时交互红线）**？**到底什么规模、什么上下文长度、什么输入输出比（`I:O Ratio`）下应该老老实实保持混部，什么阈值下必须切换为 PD 分离？** 跨节点传输数 GB 的 `KV Cache` 需要多少网络带宽（`NVLink` vs. `400G RoCEv2/IB` vs. `TCP` 手算公式）？主流开源方案（`Mooncake`、`NVIDIA Dynamo [NIXL]`、`SGLang`、`vLLM V1`）各自的架构优劣是什么？

![PD 分离 vs. 混部 Chunked Prefill 适用边界、KV Cache 带宽手算与引擎选型总结图](../../assets/svg/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

**一句话讲透 PD 分离的物理本质**：Prefill 是 **“算力密集型（Compute-Bound，算术强度 $>300\text{ FLOPs/Byte}$，需要小 Batch 吃满 TensorCore）”**，而 Decode 是 **“显存带宽密集型（Memory-Bandwidth-Bound，算术强度 $\approx 1\sim 16\text{ FLOPs/Byte}$，需要大 Batch 凑齐几十个并发请求一起读一次权重来摊薄 HBM 带宽）”**。把它们塞在同一组 GPU 上混跑，就像**在一条高速公路上同时跑超长重载火车（Prefill）和F1赛车（Decode）**——火车一进站，后面所有赛车全被堵死（`TPOT` 剧烈抖动）；而为了保赛车不堵车去限制火车长度（`Chunked Prefill`），火车进站时间又被拉长（`TTFT` 恶化且反复重读历史 KV Cache）！

| 推理部署架构层级 | 核心机制 | `16K` 输入下 `P99 TTFT` | `16K` 输入下 `P99 TPOT` (`<=25ms` 合格) | 单卡 SLA 达标有效吞吐 (`Goodput`) | 单百万 SLA 达标 Token 成本 (`$/1M`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：朴素混部连续批处理（`Prefill-First Colocated`，无分块，反模式）** | 新请求 Prefill 优先抢占正在运行的 Decode 批次 | `620 ms` | **`148.0 ms`（超标 `5.9 倍`！）** | `520 tok/s/GPU` | `$0.92 / 1M` | ❌ **严禁用于长文本在线服务**：长 Prompt 进来瞬间所有流式输出全部卡顿假死 |
| **Tier 2：混部 + 分块预填充（`Colocated + Chunked Prefill`）** | 将 16K Prompt 切成 16 个 `1024` 块，每步搭车 Decode 一起跑 | `940 ms`（分块拉长首字） | `58.5 ms`（仍超 `25ms` 红线） | `890 tok/s/GPU` | `$0.64 / 1M` | ✅ **中小集群/短文本最优**：`<=8 GPU`、输入 `<2K` 或无 RDMA 网络时的黄金标配 |
| **Tier 3：无 RDMA 的普通 TCP/IP 跨机 PD 分离（反模式）** | 虽分离了 P/D，但通过内核态 TCP 拷贝 `2.68 GB` KV | **`1,120 ms`（网络卡死 `+480ms`！）** | `21.0 ms` | `1,050 tok/s/GPU` | `$0.55 / 1M` | ❌ **得不偿失**：省下的计算时间全赔在了慢速 TCP 网卡拷贝 KV Cache 上 |
| **Tier 4：基于 `GPUDirect RDMA / NVLink` 的逐层流式 PD 分离（`Dynamo / Mooncake / SGLang`，推荐）** | 独立 P/D 池与并行策略 + 第 $L$ 层算完立即后台 RDMA 推送 KV | **`485 ms`** | **`16.4 ms`（降低 `3.57 倍`！）** | **`1,940 tok/s/GPU`（提升 `2.18 倍`）** | **`$0.34 / 1M`（省 `46.9%`）** | 🏆 **大规模长上下文金标准**：彻底消灭 TPOT 尖峰，P 与 D 各自独立采用最优并行度 |

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（原理 + 边界 + 手算公式）

### Q1：为什么混部（Colocated）即使用了 `Chunked Prefill`（Sarathi-Serve / vLLM 默认方案），在长上下文（`>=8K`）下依然打不过 PD 分离？
* **从第一性原理拆解混部的三大死结**：
  1. **显存带宽被重复读取历史 KV 浪费（Quadratic KV Reload Tax）**：当你把一个 `16,384` Tokens 的请求切成 16 个 `1,024` 的 Chunk 分 16 步执行时，第 1 步读 1K KV，第 2 步读 2K KV……第 16 步读 16K KV——前缀 KV 被从 HBM 重复读取了 $\frac{16 \times 17}{2} = 136$ 个块次（比一次性 Prefill 多耗费数倍 HBM 带宽）！
  2. **Prefill 与 Decode 最优并行方式截然相反（Parallelism Conflict）**：
     * **Prefill 池** 没有自回归依赖、激活值大但无需存长久 KV Pool，最适合 **高张量并行（`TP=8`）或上下文并行（`CP`）** 来极速压低 `TTFT`；
     * **Decode 池** 需要攒出超大 Batch（`B=64–256`）才能提高算术强度吃满 HBM，且在 MoE 模型（如 `DeepSeek-671B`）中必须用 **大专家并行（`Attention DP + Expert EP=32/64`）**。如果混部在同一实例里，P 和 D 被迫共用同一套尴尬的折中并行切分！
  3. **硬件异构配平（Heterogeneous Hardware Matching）无法在混部中实现**：PD 分离允许你用 **高算力/低显存卡（如 `H100 80GB` 或 `B200`）专攻 Prefill**，而用 **超高显存容量与高带宽卡（如 `H200 141GB`、`MI300X 192GB` 或 `TPU7x 192GB`）专攻 Decode**！

### Q2：【架构师拍板核心】到底什么时候该保持“混部 + Chunked Prefill”，什么时候必须上“PD 分离”？
* **一张表给出明确的定量切换阈值（5 大维度硬边界）**：

| 决策维度 | ✅ 坚决保持【混部 + Chunked Prefill】（不要过度设计！） | 🚀 必须切换【PD 分离（Prefill-Decode Disaggregation）】 | 背后的物理与经济学原因 |
| :--- | :--- | :--- | :--- |
| **1. 集群规模（同模型 GPU 卡数）** | **`<= 8 张 GPU`**（单机或小规模实例） | **`>= 16 张 GPU`**（至少可拆出 `1–2` 台 P 节点 + `2–4` 台 D 节点） | 卡数太少时拆分成两个池会导致各自池内弹性不足、排队率反而上升 |
| **2. 平均输入长度 ($S_{\text{in}}$)** | **短输入（$S_{\text{in}} < 2,048\text{ Tokens}$）** | **中长上下文（$S_{\text{in}} \ge 4,096\sim 128,000\text{ Tokens}$）** | 短输入 Prefill 仅需 `<30ms`，对 Decode 干扰极小；`>=4K` 后单次 Prefill 耗时显著超过 `TPOT` 阈值 |
| **3. 输入输出 Token 比（`I:O Ratio`）** | **低输入输出比（`I:O < 4 : 1`）**，如长文续写、深度思考（Reasoning 长 CoT 输出） | **高输入输出比（`I:O >= 8 : 1` 乃至 `50 : 1`）**，如企业文档 RAG、代码仓问答、多轮 Agent | `I:O` 越高，Prefill 算力占比越重，对流式 Decode 的脉冲抢占越猛烈 |
| **4. 尾延迟 SLA 严苛度** | 宽松 SLA（`P99 TPOT <= 60–100ms` 或离线批处理） | **严苛实时交互 SLA（`P99 TPOT <= 20–25ms` 且 `P99 TTFT <= 800ms`）** | 只有物理隔离 P 与 D，才能从根源上保证 Decode 步时呈一条平滑直线（零毛刺） |
| **5. 节点间网络条件** | **仅有普通 `10G/25G/100G TCP` 网卡（无 RDMA / 无 NVLink 跨机）** | **具备 `100G–800G GPUDirect RDMA (RoCEv2 / IB)`、机柜级 `NVLink` 或 `TPU ICI`** | 无 RDMA 时传输数 GB KV Cache 的耗时超过 Prefill 本身（除非是 `DeepSeek MLA` 极小 KV） |

### Q3：【手算基本功】如何精确手算 PD 分离时跨节点传输 `KV Cache` 所需的带宽与耗时？
* **三步手算公式（架构师面试与方案评审必考）**：
  1. **第 1 步：算单请求 KV Cache 字节数（$M_{\text{KV\_req}}$）**：
     $$M_{\text{KV\_req}} = 2_{\text{(K+V)}} \times L_{\text{层数}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S_{\text{in}} \times \text{Bytes}_{\text{精度}}$$
     * **算例 A（`Llama-3-70B`，GQA：$L=80, n_{\text{kv}}=8, d_h=128$，输入 $S_{\text{in}} = 16,384$ Tokens）**：
       * 若用 **`BF16`（2 Bytes）**：每 Token KV = $2 \times 80 \times 8 \times 128 \times 2 = \mathbf{327,680\text{ Bytes (320 KB/tok)}}$；`16K` 请求总 KV = $\mathbf{5.37\text{ GB}}$！
       * 若用 **`FP8`（1 Byte）**：每 Token KV = $\mathbf{160\text{ KB/tok}}$；`16K` 请求总 KV = $\mathbf{2.68\text{ GB}}$！
     * **算例 B（`DeepSeek-V3/R1 671B`，采用 `MLA` 多头潜在注意力压缩：$L=61$，每层仅缓存 $512\text{d}$ 潜在向量 + $64\text{d}$ 旋转位置 = $576$ 个元素，`FP8` 1 Byte）**：
       * 每 Token KV = $61 \times 576 \times 1 \approx \mathbf{35.1\text{ KB/tok}}$（哪怕 `BF16` 也仅 **`70.3 KB/tok`**！比 GQA 小 **`4.6 倍`**，比传统 MHA 小 **`37 倍`**！）；`16K` 请求在 `FP8` 下总 KV 仅 **`0.575 GB`**！**这就是为什么 DeepSeek-V3/R1 天然是 PD 分离的绝配！**
  2. **第 2 步：算不同网络下的裸传输耗时（$T_{\text{KV\_raw}} = \frac{M_{\text{KV\_req}}}{\text{BW}_{\text{eff}}}$）**：
     * **普通 `100G TCP/IP`（有效带宽 `~5.5 GB/s`）**：传 `2.68 GB`（70B FP8 KV）需 $2.68 / 5.5 = \mathbf{487\text{ ms}}$（完全不可接受！）；
     * **单张 `400G RoCEv2 / InfiniBand NDR` 开启 `GPUDirect RDMA`（有效带宽 `~45 GB/s`）**：传 `2.68 GB` 仅需 $2.68 / 45 = \mathbf{59.6\text{ ms}}$（若为 DeepSeek MLA `0.575 GB` 则仅需 **`12.8 ms`**！）；
     * **机内 / `GB200 NVL72` `NVLink`（有效带宽 `~750 GB/s`）**：传 `2.68 GB` 仅需 **`3.6 ms`**！
  3. **第 3 步：开启“逐层流水线推送（Layer-Wise Overlapped KV Push）”后的实际暴露延迟**：
     * 现代 PD 引擎（`Mooncake` / `Dynamo NIXL` / `SGLang`）**绝不会傻等全部 80 层都算完才开始传 KV**！而是 **Prefill 节点剛算完第 1 层，RDMA 网卡立刻在后台把第 1 层的 KV 推向 Decode 节点，同时 GPU 继续算第 2 层 Prefill**——75%–85% 的传输时间被 Prefill 自身计算完全掩盖，最终暴露在 `TTFT` 上的净网络开销仅 **`14.9 ms`**！

### Q4：当前四大主流开源 PD 分离方案（`Mooncake`、`NVIDIA Dynamo`、`SGLang`、`vLLM V1`）怎么选？
* **四大 PD 分离技术栈横向对比表**：

| 维度 | **Mooncake (`Kimi` 开源 KV 中心化架构)** | **NVIDIA Dynamo (`NIXL` + Smart Router)** | **SGLang (`Disaggregated Serving` + `Mooncake/NIXL`)** | **vLLM V1 (`PyNccl / Mooncake / NIXL Connector`)** |
| :--- | :--- | :--- | :--- | :--- |
| **核心架构理念** | **以分布式 KV Cache 内存池（Conductor + Transfer Engine）为中心**，统一管 DRAM/NVMe/RDMA | NVIDIA 官方全栈推理控制面，底层基于 **`NIXL`（统一 NVLink/IB/RoCE/SSD 零拷贝传输库）** | 原生 **`RadixAttention` 前缀树路由** + 极简 P/D Worker 直连 RDMA 队列 | 模块化 **`KVConnector` 插件接口**，广泛兼容多硬件（GPU/TPU/昇腾） |
| **最大杀手锏** | 多机 CPU DRAM 组建 TB 级全局共享 KV 池，超长多轮命中率极高 | 与 TensorRT-LLM / vLLM / SGLang 皆可对接，自带 KV 感知动态扩缩容规划器 | `DeepSeek-V3/R1`（MLA + DeepEP + 双微批重叠）与多轮 Agent 性能之王 | 生态最广、模型 Day-0 支持最快、API 最标准 |
| **首选适用场景** | 超长上下文多轮对话、高前缀复用率的百卡级公有/私有 MaaS | 标准 NVIDIA `H100/H200/GB200` 生产集群统一编排 | **首选用于 `DeepSeek-671B`、重度 RAG 与多智能体生产推理** | 通用多模型企业中台与跨异构芯片（含 `vLLM-TPU`）标准化底座 |

---

## 3. 🧭 直观可视化图解（Mermaid 对比图 + 决策树 + 时序图）

### 3.1 为什么混部会卡顿 vs. PD 分离如何实现零干扰流水线

```mermaid
flowchart TB
    subgraph Coloc["⛔ 混部模式 (Colocated): 16K Prefill 插入导致所有正在吐字的 Decode 用户卡死 60~148ms"]
        direction LR
        D1["Decode Step #1\n(16ms 正常吐字)"] --> D2["Decode Step #2\n(16ms 正常吐字)"]
        D2 --> P_IN["🚨 突发 16K 长 Prompt 进入!\nPrefill 霸占 TensorCore 140ms\n(64 个在线用户屏幕瞬间卡顿假死!)"]
        P_IN --> D3["Decode Step #3\n(P99 TPOT 飙至 148ms!)"]
    end

    subgraph Disagg["🏆 PD 分离模式 (Disaggregated): Prefill 池与 Decode 池物理隔离 + 逐层 RDMA 零拷贝推送"]
        direction LR
        PP["Prefill 专属池 (TP=8 高算力)\n全速吞吐 16K Prompt (470ms)"] ==>|"算完第 L 层立即通过 400G GPUDirect RDMA\n推送 KV 块 (仅 14.9ms 暴露开销)"| DP["Decode 专属池 (大 Batch + EP/DP)\n全程 16.4ms 匀速丝滑吐字 (零抖动!)"]
    end
```

### 3.2 混部（Colocated + Chunked Prefill）vs. PD 分离选型决策树

```mermaid
flowchart TD
    A["企业在线大模型推理架构选型"] --> B{"同模型部署规模是否 >= 16 张 GPU/TPU?"}
    B -- "否 (<=8 卡小池)" --> C["✅ 保持混部 + Chunked Prefill\n(vLLM / SGLang 开启 chunked_prefill_size=1024)"]
    B -- "是 (>=16 卡中大型池)" --> D{"业务平均输入长度与输入输出比 (I:O Ratio)?"}
    D -- "短输入 (<2K Tokens) 且 I:O < 4:1" --> C
    D -- "中长输入 (>=4K Tokens) 或 I:O >= 8:1 (RAG/Agent/Code)" --> E{"节点间是否具备 GPUDirect RDMA (RoCEv2/IB >=100G) 或采用 MLA 模型?"}
    E -- "无 RDMA 且为传统大 KV 模型 (如 70B BF16)" --> F["⚠️ 保持混部 + Chunked Prefill + 开启 FP8 KV 压缩\n(避免慢速 TCP 传输数 GB KV 拖垮 TTFT)"]
    E -- "具备 100G~800G RDMA / NVLink 或采用 DeepSeek MLA" --> G["🏆 启用 PD 分离架构 (Tier 4: SGLang / Dynamo / Mooncake)\n按流量负载动态配平 xP : yD 节点比 (如 3P : 5D)"]
```

### 3.3 全链路 Prefix Cache 路由、逐层 GPUDirect RDMA KV 传输与解码时序图

```mermaid
sequenceDiagram
    autonumber
    participant Client as 终端用户 / Agent
    participant Router as KV-Aware 智能路由网关 (Dynamo / Mooncake)
    participant PWorker as Prefill 实例 (TP=8)
    participant NIC as 400G ConnectX-7 (GPUDirect RDMA)
    participant DWorker as Decode 实例 (大 Batch 高带宽池)

    Client->>Router: 提交 `16,384 Tokens` 请求
    Router->>Router: 查询全局 Prefix 索引 -> 命中 `PWorker #2` 上的 `6,000 Tokens` 缓存，同时在 `DWorker #5` 预分配分页 KV 物理槽位描述符
    Router->>PWorker: 转发请求 + `DWorker #5` 远端 HBM 内存 RDMA 句柄 (`rkey / vaddr`)
    loop 第 L = 1 .. 80 层前向计算与传输流水线重叠
        PWorker->>PWorker: TensorCore 计算第 $L$ 层新增 `10,384 Tokens` 的 K, V 张量
        PWorker->>NIC: 触发异步 `IBV_WR_RDMA_WRITE`: 零拷贝从 P 卡 HBM 直写 `DWorker #5` 的 HBM 分页槽位 (与第 $L+1$ 层计算并行!)
    end
    PWorker-->>Client: 第 80 层算完立即流式吐出第 1 个 Token (`P99 TTFT = 485ms`)
    NIC->>DWorker: 最后一层 KV 到达完成通知 (`+14.9ms`)
    loop 第 2 .. 512 个输出 Token
        DWorker-->>Client: 在纯净无干扰的 Decode 池中以 `P99 TPOT = 16.4ms` 稳定流式输出
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：朴素混部（Prefill 优先，无分块） | Tier 2：混部 + `Chunked Prefill` (`chunk=1024`) | Tier 3：基于普通 TCP/IP 的跨机 PD 分离 | Tier 4：基于 `400G GPUDirect RDMA` 逐层流式 PD 分离（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **`16K` 输入下 `P99 Decode TPOT`** | **`148.0 ms`（严重卡顿超标）** | `58.5 ms`（仍超出 `25ms` SLA） | `21.0 ms` | **`16.4 ms`（较 Chunked 混部降低 `3.57 倍`！）** |
| **`16K` 输入下 `P99 Prefill TTFT`** | `620.0 ms` | `940.0 ms`（分 16 步搭车拉长首字） | **`1,120.0 ms`（TCP 拷 KV 卡死 `+480ms`）** | **`485.0 ms`（算力吃满 + 逐层 RDMA 重叠）** |
| **单卡 SLA 达标有效吞吐 (`Goodput`)** | `520 tok/s/GPU` | `890 tok/s/GPU` | `1,050 tok/s/GPU` | **`1,940 tok/s/GPU`（较 Tier 2 提升 `2.18 倍`！）** |
| **P 与 D 独立并行度与异构选卡** | ❌ 强制绑定同配置 | ❌ 强制绑定同配置 | ✅ 可分离但受限于网络 | **✅ P 池高 `TP` 选算力卡，D 池高 `EP/DP` 选大显存带宽卡** |
| **单百万 SLA 达标 Token 成本 (`$/1M`)** | `$0.92 / 1M` | `$0.64 / 1M` | `$0.55 / 1M` | **`$0.34 / 1M`（较 Tier 2 节省 `46.9%`）** |

---

## 5. 🧪 容量/成本拐点算例与交互式 H5 计算器

### 5.1 基准算例假设（`Llama-3-70B-FP8` & `DeepSeek-671B` / `16K` 输入 + `512` 输出）
* **业务规模**：企业知识库 RAG 与长上下文 Agent 场景，平均输入 `16,384` Tokens，平均输出 `512` Tokens（`I:O = 32:1`），SLA 要求 `P99 TTFT <= 800ms` 且 `P99 TPOT <= 25ms`。
* **工程实测与仿真代码**：见 [`experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/run.py`](../../experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/run.py) 与 [`results.json`](../../experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式 H5 边界与 KV 带宽计算器**：📊 [打开 PD 分离 vs. 混部边界与 KV 传输耗时计算器 (`prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-sizer.html`)](../reports/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-cards.html`)](../reports/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Naive Colocated", "p99_tpot_ms": 148.0, "p99_ttft_ms": 620.0, "goodput_tok_s": 520, "cost_per_1m_usd": 0.92 },
      { "tier": "Tier 2: Colocated + Chunked Prefill", "p99_tpot_ms": 58.5, "p99_ttft_ms": 940.0, "goodput_tok_s": 890, "cost_per_1m_usd": 0.64 },
      { "tier": "Tier 3: PD Split Over TCP/IP", "p99_tpot_ms": 21.0, "p99_ttft_ms": 1120.0, "goodput_tok_s": 1050, "cost_per_1m_usd": 0.55 },
      { "tier": "Tier 4: 400G RDMA Layer-Wise PD Split", "p99_tpot_ms": 16.4, "p99_ttft_ms": 485.0, "goodput_tok_s": 1940, "cost_per_1m_usd": 0.34 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "p99_tpot_ms": "Quantity",
    "p99_ttft_ms": "Quantity",
    "goodput_tok_s": "Quantity",
    "cost_per_1m_usd": "Price"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "400G RDMA PD Disaggregation Cuts P99 TPOT 3.57x (16.4ms) and Lifts SLA Goodput 2.18x (1,940 tok/s/GPU)",
    "subtitle": "SLA-compliant goodput (tokens/sec/GPU) across colocated and disaggregated serving tiers on 16K prompts",
    "encodings": {
      "x": "tier",
      "y": "goodput_tok_s",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **五维准入边界核验（避免在 `<8 GPU` 或无 RDMA 环境强行上 PD 分离）**：是否已核实集群满足 `>=16 GPU`、平均输入 `>=4K`（或 `I:O >= 8:1`）、且具备 `GPUDirect RDMA (RoCEv2/IB >= 100G)` 或采用 `MLA` 压缩模型？
- [ ] **开启 `FP8 KV Cache` 与逐层重叠传输（`Layer-Wise Overlapped KV Push`）**：是否已将跨节点传输的 KV Cache 量化为 `FP8`（体积减半），并启用算完第 $L$ 层立即触发 `RDMA WRITE` 的流水线重叠模式？
- [ ] **动态 P:D 实例配比弹性伸缩（`Dynamic xP : yD Autoscaling`）**：控制面（如 `NVIDIA Dynamo Planner`）是否根据实时流量的输入/输出长度比动态调整 Prefill 节点与 Decode 节点的数量配比（例如早高峰读文档 `4P:4D`，下午写代码 `2P:6D`）？
- [ ] **Decode 侧预先分配 KV 页表槽位（`Pre-Allocated Paged KV Descriptors`）**：是否在 Prefill 开始前就由 Decode 节点返回目标 HBM 页表虚拟地址，避免 Prefill 算完后因 Decode 显存满而阻塞等待？
- [ ] **与全局 Prefix Cache 池（`Mooncake / RadixAttention`）联动**：当第 2 轮对话命中全局 DRAM/RDMA 前缀缓存池时，是否直接跳过重复 Prefill、僅计算增量 Tokens？
