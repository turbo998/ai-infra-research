# Google Cloud TPU 在线推理服务引擎选型：`vLLM-TPU V1 (Unified PyTorch/JAX)` vs. `JetStream / MaxEngine`、持久化 `XLA` 分桶缓存、`Pallas Ragged PagedAttention` 与 `ICI` 跨机 P/D 分离架构指南

> **适用读者**：负责在 Google Cloud TPU（`v6e Trillium`、`TPU7x Ironwood`、`TPU 8i`、`v5e`）上构建高并发、低延迟大模型在线推理与多智能体（Agentic RAG）服务底座的 Cloud & AI Infrastructure Solution Architect、推理系统研发负责人与平台 SRE。
> **核心课题**：当企业将一个平均输入 `8,192` Tokens、输出 `512` Tokens、要求 **`P99 TPOT <= 25ms`** 的生产级 Agent 在线服务部署到 **`64` 颗 TPU 芯片（`v6e Trillium / TPU7x Ironwood`）** 上时，为什么直接用“未分桶的 `HuggingFace / PyTorch-XLA` 连续 KV 缓存推理（Tier 1）”或“混部未分离的 `JetStream + MaxEngine`（Tier 2）”会导致 **线上每逢新长度请求就卡死 `45 秒` 触发 XLA 重编译、连续 KV 槽位浪费 `68%` 显存、或因长 Prompt Prefill 抢占 `MXU` 把 Decode P99 TPOT 拉爆到 `42ms–115ms`（达标单百万 Token 成本高达 `$1.65/1M`）**？如何通过 **`vLLM-TPU V1` 统一后端 + `Hyperdisk ML` 持久化 `XLA` 编译缓存（`<28s` 极速扩容，运行期 `0` 重编译）+ `Pallas Ragged PagedAttention`（异步 `HBM -> VMEM` DMA 预取，Decode `MBU` 达 `79.4%`）+ Slice 内 `1.2 Tbps ICI` 预填充/解码（P/D）物理分离** 四级架构，将 P99 TPOT 压降至 **`14.8ms`**、单百万 SLA 达标 Token 成本压至 **`$0.21 / 1M`**，并将月度 64 芯片推理池 TCO 降低 **`73.3%`（`$420,000 -> $112,000/月`）**？

![Google Cloud TPU 在线推理引擎选型：vLLM-TPU V1 vs. JetStream、Pallas PagedAttention 与 ICI P/D 分离架构总结图](../../assets/svg/tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在 TPU（`v6e Trillium / TPU7x Ironwood / TPU 8i`）上跑高并发在线推理，**架构师必须解决两个与 GPU 截然不同的底层挑战：① 如何让静态图编译器 `XLA` 在面对千变万化的用户输入长度时实现“零运行期重编译”；② 如何利用 TPU 独特的片上高速暂存器（`VMEM`）与自定义内核语言 `Pallas`，让非连续分页的 `PagedAttention` 跑满 `79%+` 的 HBM 显存带宽（MBU）**：

| TPU 推理架构层级 | 适用场景 | P99 Decode `TPOT` (`<=25ms` SLA) | Decode 显存带宽利用率 (`MBU`) | 单百万 SLA 达标 Token 成本 (`$/1M`) | 月度 `64` 芯片等效推理总成本 (`TCO`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：未分桶 `HuggingFace / PyTorch-XLA` + 连续静态 KV 槽位（反模式）** | ❌ **严禁用于任何在线推理服务** | **`115.0 ms`（重编译卡死！）** | **`22.0%`** | `$1.65 / 1M` | `$420,000 / 月` | ❌ **严禁上生产**：新长度请求触发 45s 编译停服，68% 显存被空白 Padding 吞噬 |
| **Tier 2：分桶 `JetStream + MaxEngine`（Prefill/Decode 同池混部，无前缀缓存）** | 纯 `JAX / MaxText` 训练回路内部的 RLHF/GRPO 批量采样（Rollout） | `42.0 ms`（被 8K Prefill 阻塞） | `54.0%` | `$0.58 / 1M` | `$245,000 / 月` | ✅ **RL 采样专用**：免权重转换直连 `MaxText`，但生态功能少于 `vLLM` |
| **Tier 3：同池混部 `vLLM-TPU V1`（`Pallas PagedAttention + Chunked Prefill`）** | `<=16` 芯片中小规模或短 Prompt 问答服务 | `24.5 ms` | `66.5%` | `$0.36 / 1M` | `$178,000 / 月` | ✅ **中小池标配**：完整兼容 OpenAI API 与 Prefix Cache，但 32K 突发仍抖动 |
| **Tier 4：`vLLM-TPU V1` + 持久化 `XLA` 分桶缓存 + `Pallas Ragged PA` + `ICI` 跨机 P/D 分离（推荐）** | **企业级高并发 Agent、长上下文 RAG 与多 LoRA 生产主力** | **`14.8 ms`（稳守 SLA）** | **`79.4%`** | **`$0.21 / 1M`（省 `87.3%`）** | **`$112,000 / 月`（省 `73.3%`）** | 🏆 **生产金标准**：运行期 `0` 重编译，`VMEM` 异步预取吃满 HBM，`ICI` 6ms 传 KV |

### 💡 架构师的三大黄金铁律
1. **引擎选型一锤定音：面向外部/企业业务的在线微服务统一选 `vLLM-TPU V1`，面向 `MaxText` 强化学习训练内部采样选 `JetStream + MaxEngine`**：
   * 过去 TPU 推理存在 `PyTorch/XLA` 与 `JAX (JetStream)` 生态割裂的问题。最新的 **`vLLM V1 TPU Unified Backend`（`vllm-tpu`）** 已将前端调度（OpenAI API、自动前缀缓存 `Automatic Prefix Caching`、`Chunked Prefill`、`XGrammar` 结构化 JSON 解码、动态 `Multi-LoRA`）与底层极致 **`JAX / XLA / Pallas` 算子** 合二为一——架构师无需手写一行 JAX 代码，直接传一个 HuggingFace 模型名即可在 `v6e Trillium` 或 `TPU7x Ironwood` 上跑出极致吞吐！
   * 只有当团队正在用纯 JAX 的 `MaxText` 跑 RLHF/GRPO 强化学习、需要在内存里零转换直接调用 Actor 模型做批量采样时，才选用 **`JetStream + MaxEngine`**。
2. **用“指数级 `(num_tokens, batch_size)` 静态分桶 + `Hyperdisk ML` 共享持久化 `XLA` 编译缓存”彻底消灭运行期 `45 秒` 编译抖动**：
   * `vLLM-TPU` 将连续批处理（Continuous Batching）中的有效 Token 总数向上对齐至预定义的幂次桶（如 `256, 512, 1024, 2048, 4096, 8192, 16384, 32768`），并在首个 Pod 制作时将全部桶编译出的 `XLA HLO` 二进制文件持久化保存在只读共享的 **`Hyperdisk ML`** 上。当 `KEDA` 触发自动扩容拉起新 Pod 时，新副本直接秒级加载已编译好的 XLA 缓存（**`<28 秒` 就绪，运行期 `xla_compilation_cache_misses == 0`**）！
3. **用“`Pallas Ragged PagedAttention`（片上 `VMEM` 异步 DMA）+ Slice 内 `1.2 Tbps ICI` 预填充/解码（P/D）分离”压榨 `79.4%` HBM 带宽**：
   * TPU 没有 GPU 那样的硬件级 L1/L2 Cache 自动行合并机制，如果直接从 HBM 随机读取散落的 KV 分页，带宽利用率只有 `22%`。通过 **`Pallas`（JAX 的底层内核语言，类似 Triton）** 编写的 **`Ragged PagedAttention`**，显式驱动硬件异步 DMA 引擎提前将未来 4 个块的 KV Page（`block_size = 16/32`）搬入片上高速 **`VMEM` 暂存器**，与 `MXU` 的 $Q K^T$ 矩阵乘完全重叠！
   * 同时，在同一个 2D/3D Torus Slice 内部，将计算密集型 **Prefill 节点** 与带宽密集型 **Decode 节点** 角色分离：Prefill 算完后的 KV Cache 直接走芯片间 **`1.2 Tbps ICI` 光链路**（`32K` 上下文 KV 传输耗时仅 **`<6ms`**，完全绕过慢速主机 TCP/IP），将 Decode P99 `TPOT` 稳定锁在 **`14.8ms`**！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：为什么在 GPU 上直接用 `Triton` 写的 `PagedAttention` 搬到 TPU 上跑不快，非要用 `Pallas Ragged PagedAttention`？
* **硬件访存粒度差异：GPU 靠 `32-Thread Warp Coalescing`，TPU 靠 `异步块级 DMA (HBM -> VMEM)`**：
  * GPU 的 `SM` 允许 32 个线程按 `128 Byte` 缓存行粒度去 HBM 零散抓数据；而 TPU 的 `MXU / VPU` 只能直接访问片上的 **`VMEM`（Vector Memory，数十 MB 的软件管理显式 SRAM）**，所有从 `HBM` 到 `VMEM` 的数据搬运必须通过 **块级异步 DMA 指令（`pltpu.async_copy`）** 发起。
  * 如果每次只搬 1 个 Token 的 KV，DMA 启动开销会把流水线拖死；**`Pallas Ragged PagedAttention`** 将非连续请求打包成紧凑的变长序列（Ragged Tensor），以 `16` 或 `32` Tokens 为一个物理 Page 块发起双缓冲异步 DMA，让 `HBM -> VMEM` 搬运耗时 `100%` 藏在 `MXU` 计算背后，将有效显存带宽利用率（`MBU`）从 `22.0%` 推升至 **`79.4%`**！

### Q2：在 TPU 上做在线推理，什么时候该选 `v6e (Trillium)`，什么时候该选 `TPU7x (Ironwood)` 或 `TPU 8i`？
* **按模型参数量与 HBM 容量/带宽拐点拍板**：
  * **选 `TPU v6e (Trillium, 32GB HBM @ 1.64 TB/s)`**：服务 `8B–32B` 模型，或经 `INT8/FP8` 量化后的 `70B` 模型（`tp=4` 或 `8` 单主机部署），性价比极高，是常规企业 RAG 与代码补全的主力机型。
  * **选 `TPU7x (Ironwood, 192GB HBM3e @ 7.37 TB/s)` 或推理专用 `TPU 8i`**：服务满血 `70B–671B` 超大稀疏 MoE 模型、需要维持 `64K–1M` 超长上下文 `KV Cache` 池、或需要 `4x SparseCores` 加速稀疏注意力与动态专家路由的旗舰推理场景——单卡 `192GB` 显存与 `7.37 TB/s` 带宽可大幅缩减所需的 `TP` 切分卡数！

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 端到端参考架构 + 时序图）

### 3.1 Google Cloud TPU 在线推理引擎与拓扑选型决策树

```mermaid
flowchart TD
    A["Google Cloud TPU 推理服务架构选型需求"] --> B{"调用方与业务场景类型?"}
    B -- "MaxText 纯 JAX 强化学习 (RLHF/GRPO) 内部在线 Rollout 采样" --> C["选用 JetStream + MaxEngine\n(零权重转换直接共享 MaxText 分片与 Orbax 内存权重)"]
    B -- "企业生产在线服务 (OpenAI API / Agent / RAG / Multi-LoRA)" --> D["统一选用 vLLM-TPU V1 (Unified Backend)"]

    D --> E{"输入 Prompt 长度分布与 P99 TPOT SLA 要求?"}
    E -- "短文本 (<=4K) 或小规模池 (<=16 芯片)" --> F["Tier 3: 同池混部 vLLM-TPU V1\n开启 Chunked Prefill + Prefix Caching + Pallas PagedAttention"]
    E -- "长上下文 Agent/RAG (8K~128K) 且要求 P99 TPOT <= 20ms" --> G["Tier 4: Slice 内 ICI 预填充/解码 (P/D) 物理分离架构\n① Hyperdisk ML 持久化 XLA 编译缓存 (0 重编译)\n② Prefill 节点算完经 1.2 Tbps ICI (<6ms) 直传 KV 至 Decode 节点"]
```

### 3.2 Tier 4 `vLLM-TPU V1` + 持久化 `XLA` 缓存 + `Pallas` + `ICI P/D` 分离架构图

```mermaid
flowchart LR
    subgraph Cache["⚡ 1. 启动期零编译加速层 (<28s 扩容)"]
        HD["Hyperdisk ML 共享只读盘\n预存全分桶 (256..32K) AOT XLA HLO 二进制"] ==> PRE & DEC
    end

    subgraph Slice["🕸️ 2. 同 Slice 内部 ICI 预填充/解码分离池 (64 芯片)"]
        GW["vLLM Prefix-Aware 路由网关\n(按共享系统提示词哈希亲和)"] --> PRE["Prefill 节点池 (Compute-Bound)\nChunked Prefill + Prefix Cache 命中"]
        PRE ==>|"1.2 Tbps 芯片间 ICI 直传 KV 块\n(32K KV 耗时 < 6ms，零主机 TCP 开销)"| DEC["Decode 节点池 (Bandwidth-Bound)\nPallas Ragged PagedAttention"]
    end

    subgraph Kernel["🔬 3. TPU 芯片内 Pallas 双缓冲流水线 (79.4% MBU)"]
        DEC --> DMA["异步块级 DMA (HBM -> 片上 VMEM)\n预取下 4 个 16-Token KV Pages"]
        DMA <==>|"100% 重叠执行"| MXU["TensorCore MXU (QK^T & SV)\nP99 TPOT = 14.8ms ($0.21/1M)"]
    end
```

### 3.3 全链路前缀缓存命中、`ICI` 跨机 KV 传输与 `Pallas VMEM` 解码时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as 企业 Agent 应用
    participant Router as vLLM-TPU P/D 路由器
    participant PNode as TPU Prefill 节点 (Host #0..#3)
    participant DNode as TPU Decode 节点 (Host #4..#15)

    App->>Router: 发送 `12,000 Tokens` 多轮 Agent 请求 (前 `8,000 Tokens` 为共享工具定义与历史)
    Router->>PNode: 路由至已缓存该会话前缀的 Prefill 节点; 填充剩余 `4,000` 新 Tokens 至 `4096` 静态桶 (`0` XLA 重编译!)
    PNode->>PNode: 命中 `8,000 Tokens` Prefix Cache，仅对新增 `4,096` 桶执行 `MXU` Prefill (`TTFT = 285ms`)
    PNode->>DNode: 通过 Slice 内部 `1.2 Tbps ICI` 光链路直接 DMA 推送 KV Pages (`4.2ms` 完成传输)
    loop 自回归生成每个输出 Token (P99 TPOT = 14.8ms)
        DNode->>DNode: `Pallas Ragged PagedAttention` 异步将 HBM 中的 16-Token KV 块预取至片上 `VMEM` 并与 `MXU` 计算重叠 (`79.4%` MBU)
        DNode-->>App: 流式输出 SSE Token
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：未分桶 `HF/PyTorch-XLA` + 连续 KV | Tier 2：分桶 `JetStream + MaxEngine`（同池混部） | Tier 3：同池混部 `vLLM-TPU V1` (`Chunked Prefill`) | Tier 4：`vLLM-TPU V1` + `Pallas Ragged PA` + `ICI P/D` 分离（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **P99 Decode `TPOT` 尾延迟** | **`115.0 ms`（频繁触发 45s 重编译停顿）** | `42.0 ms`（被长 Prefill 抢占阻塞） | `24.5 ms` | **`14.8 ms`（100% 满足 `<=25ms` 严苛实时 SLA）** |
| **Decode 显存带宽利用率 (`MBU`)** | **`22.0%`（连续槽位碎片 + 无 VMEM 预取）** | `54.0%` | `66.5%` | **`79.4%`（`Pallas` 双缓冲 `HBM->VMEM` 异步 DMA）** |
| **KEDA 新副本扩容冷启动耗时** | `420 秒`（现场从头编译全部 XLA 图） | `140 秒` | `45 秒` | **`<28 秒`（直接挂载 `Hyperdisk ML` 预编译 XLA 缓存）** |
| **企业级特性（Prefix Cache / LoRA / JSON）** | ❌ 缺失 | ⚠️ 基础支持 | ✅ 完整支持 | **✅ 原生支持 Prefix Cache、Multi-LoRA 与 `XGrammar`** |
| **单百万 SLA 达标 Token 成本 (`$/1M`)** | `$1.65 / 1M` | `$0.58 / 1M` | `$0.36 / 1M` | **`$0.21 / 1M`（较 Tier 1 降低 `87.3%`）** |
| **月度 `64` 芯片等效推理总 TCO** | `$420,000 / 月` | `$245,000 / 月` | `$178,000 / 月` | **`$112,000 / 月`（节省 `73.3%`）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（`64` 芯片 `TPU v6e Trillium / TPU7x` 在线推理切片 / `8K` 输入 + `512` 输出）
* **业务规模**：企业部署 `64` 颗 TPU 芯片承载多轮 Agent 与 RAG 生产流量（平均输入 `8,192` Tokens、共享前缀命中率 `65%`、平均输出 `512` Tokens），要求 `P99 TPOT <= 25ms`。
* **工程实测与仿真代码**：见 [`experiments/tpu_inference_vllm_jetstream_pallas_pagedattention_pd_disaggregation_selection/run.py`](../../experiments/tpu_inference_vllm_jetstream_pallas_pagedattention_pd_disaggregation_selection/run.py) 与 [`results.json`](../../experiments/tpu_inference_vllm_jetstream_pallas_pagedattention_pd_disaggregation_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开 Google Cloud TPU 在线推理 (`vLLM-TPU` & `ICI` P/D 分离) 容量计算器 (`tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-sizer.html`)](../reports/tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-selection-cards.html`)](../reports/tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unbucketed HF + Contiguous KV", "p99_tpot_ms": 115.0, "mbu_pct": 22.0, "token_cost_usd": 1.65, "monthly_tco_k": 420.0 },
      { "tier": "Tier 2: Bucketed JetStream Co-Located", "p99_tpot_ms": 42.0, "mbu_pct": 54.0, "token_cost_usd": 0.58, "monthly_tco_k": 245.0 },
      { "tier": "Tier 3: Co-Located vLLM-TPU V1", "p99_tpot_ms": 24.5, "mbu_pct": 66.5, "token_cost_usd": 0.36, "monthly_tco_k": 178.0 },
      { "tier": "Tier 4: vLLM-TPU V1 + Pallas PA + ICI P/D Split", "p99_tpot_ms": 14.8, "mbu_pct": 79.4, "token_cost_usd": 0.21, "monthly_tco_k": 112.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "p99_tpot_ms": "Quantity",
    "mbu_pct": "Percentage",
    "token_cost_usd": "Price",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 14.8ms P99 TPOT, 79.4% Decode MBU, and Cuts Monthly TPU Serving TCO by 73.3%",
    "subtitle": "Monthly 64-chip equivalent SLA-compliant serving TCO ($ USD Thousands)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_k",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **持久化 `XLA` 编译缓存（`VLLM_XLA_CACHE_PATH`）与启动期全桶预热**：是否已将预编译好的全部 `(num_tokens, batch_size)` 幂次桶 HLO 缓存打包挂载在只读 `Hyperdisk ML` 上，并配置 Kubernetes `readinessProbe` 在预热全部完成后才接入流量（确保线上 `0` 重编译）？
- [ ] **`Pallas Ragged PagedAttention` 与 `FP8 / INT8 KV Cache` 量化开启**：是否已启用基于 `Pallas` 的 `Ragged PagedAttention` 内核与 `FP8 KV Cache`，并在 `XProf` 中确认 Decode 阶段 `HBM MBU >= 75%`？
- [ ] **长上下文场景 Slice 内 `ICI` 预填充/解码（P/D）分离**：对于平均输入 `>=4K` Tokens 且要求 `P99 TPOT <= 25ms` 的在线业务，是否已在同一 TPU Slice 内划分独立 Prefill 与 Decode 节点并通过 `ICI` 直接传输 KV 块？
- [ ] **前缀亲和路由（`Prefix-Aware Load Balancing`）**：网关层是否按 `System Prompt + 工具定义 / RAG 文档集` 的哈希将同一会话亲和路由至相同 Prefill 实例，吃满 `vLLM-TPU` 自动前缀缓存？
- [ ] **主机 CPU Tokenization 与 Detokenization 异步解耦**：是否已确认分词、JSON 语法掩码生成（`XGrammar`）与 SSE 流式组装运行在独立 CPU 进程池，零阻塞 TPU `libtpu` 主分发线程？
