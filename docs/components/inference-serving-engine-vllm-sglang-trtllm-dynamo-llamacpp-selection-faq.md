# 主流大模型推理服务引擎架构级横向选型对比：`vLLM V1` vs. `SGLang (RadixAttention)` vs. `TensorRT-LLM` vs. `NVIDIA Dynamo` vs. `Ollama / llama.cpp` 适用场景、硬件匹配与运维负担指南

> **语言 / Language**: **中文 (当前)** | [English](./inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与企业 AI 平台负责人的五大主流推理引擎/控制面（`vLLM V1`、`SGLang`、`TensorRT-LLM`、`NVIDIA Dynamo`、`Ollama/llama.cpp`）架构机制、适用业务场景、硬件车型与团队运维成本选型白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业推理技术栈选型）

![主流推理服务引擎 vLLM V1 vs SGLang vs TensorRT-LLM vs Dynamo vs Ollama 选型总结图](../../assets/svg/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**推理服务引擎没有“一招鲜”，架构师必须根据“业务形态 × 硬件车型 × 团队工程能力”精准匹配：① `Ollama / llama.cpp` 仅适合个人笔记本、断网边缘端与单用户开发沙箱（`并发 <= 4`），严禁用于数据中心高并发生产（无分布式 PagedAttention 与 NVLink EP，64 并发下吞吐跌至 `290 tok/s/GPU`）；② `vLLM V1` 是企业多模型共享中台（MaaS）的最佳默认基座（Day-0 支持 100+ 模型，横跨 `NVIDIA / AMD MI300X / Google TPU`）；③ `SGLang` 是大规模 MoE（`DeepSeek-V3/R1 EP=64 + DeepEP + 双批次重叠`）与多轮树状 Agent / 结构化 JSON 的性能冠军；④ `TensorRT-LLM` 适合固定单一旗舰模型在 `Blackwell B200 (NVFP4)` 上压榨最后 `10%~15%` 极限低延迟；⑤ 集群级生产底座应在上层部署 **`NVIDIA Dynamo`（或 `Mooncake / llm-d`）分布式控制面**，通过 **KV-Aware 缓存感知路由 + `NIXL` RDMA 跨机 PD 分离** 统一调度下层的 `SGLang / vLLM V1 / TRT-LLM` Worker，实现 `2,490 Goodput tok/s/GPU` 与 `92.8%` 多轮缓存命中率！**

- **交互式推理引擎选型匹配器**：[打开交互式计算器 (`inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-sizer.html`)](../reports/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-cards.html`)](../reports/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：`vLLM V1`、`SGLang`、`TensorRT-LLM` 在架构设计哲学与优势场景上有什么核心差异？
1. **`vLLM V1`（通用多模型企业中台的标准答案）**：
   - **架构亮点**：V1 架构重构了多进程零开销 CPU 调度器（将 Python 调度与 GPU 执行彻底解耦）、默认集成 `FlashInfer`、分段 `Piece-wise CUDA Graph`、原生 Chunked Prefill 与前缀缓存；
   - **最大不可替代性**：**生态覆盖面与异构硬件兼容性第一**（新模型发布当天 Day-0 支持，且一套代码同时支持 NVIDIA GPU、AMD ROCm、Google Cloud TPU 与 AWS Neuron）。适合需要托管 20+ 种 Dense/VLM 模型的企业内部 MaaS 平台。
2. **`SGLang`（多轮 Agent、JSON 约束解码与大型 MoE 的吞吐王者）**：
   - **架构亮点**：① 首创 **`RadixAttention`（基数树 LRU 缓存）**，把所有历史多轮对话、Few-shot 示例和 Agent 树状分支（Tree-of-Thought）自动维护在 GPU/CPU 基数树中；② 深度协同 **`XGrammar`** 实现零 CPU 阻塞的 JSON Schema 约束解码；③ 在 **`DeepSeek-V3/R1 671B` 大规模 MoE** 上率先落地完整的 `Attention DP + DeepEP (Normal/IBGDA) + Two-Batch Ping-Pong Overlap + MTP`！
3. **`TensorRT-LLM`（NVIDIA 官方 AOT 静态编译极限性能引擎）**：
   - **架构亮点**：采用 Ahead-of-Time（AOT）将模型编译为高度融合的 C++ TensorRT `.engine`，在新一代硬件（如 `Blackwell B200 NVFP4`、`GB200 NVL72`）上总能比纯 Python 框架早一步吃满底层硬件特性，单节点极限 TPOT 比通用引擎再快 `8% ~ 15%`；
   - **代价**：编译出的 `.engine` 与具体 GPU 型号、`TP/PP` 切分度数及最大 Batch/序列长度强绑定，升级版本或换模型需要重新编译，工程运维负担较高。

### Q2：`NVIDIA Dynamo` 是第 4 个推理引擎吗？它与 `vLLM / SGLang / TensorRT-LLM` 是什么关系？
- **定位厘清**：**`NVIDIA Dynamo` 不是替代 `vLLM / SGLang / TRT-LLM` 的单机算子引擎，而是架在它们之上的“集群级分布式推理控制面与编排层”**！
- **三大核心能力**：
  1. **KV-Aware Router（全局前缀缓存感知路由器）**：实时维护集群内所有 `vLLM / SGLang / TRT-LLM` 节点的 Radix KV 块索引，按公式 $\text{Cost}(i) = \text{Uncached\_Prefill\_Tokens}(i) + \lambda \cdot \text{Queue\_Load}(i)$ 路由请求，使跨机多轮 KV 命中率从 `76.4%` 飙升至 **`92.8%`**；
  2. **Disaggregated Serving Planner（动态 PD 分离弹性规划器）**：根据实时输入输出比（I:O Ratio）动态调整 Prefill 节点与 Decode 节点的配比；
  3. **`NIXL`（NVIDIA Inference Xfer Library）**：提供统一零拷贝非阻塞传输抽象，在 `NVLink`、`GPUDirect RDMA (IB/RoCEv2)`、`Host DRAM` 与 `NVMe GDS` 之间极速搬运 KV Cache。

---

## 3. 🧭 直观可视化图解（Mermaid 引擎选型决策树 + Dynamo 分布式架构）

### 3.1 企业大模型推理引擎五选一架构决策树
```mermaid
flowchart TD
    A["企业推理服务引擎选型"] --> B{"部署环境与并发规模？"}
    B -- "个人笔记本 / Mac / 断网边缘盒<br/>(并发 <= 4, CPU 或单张消费卡)" --> C["选 Ollama / llama.cpp (GGUF)<br/>• 零依赖单二进制极简部署<br/>• ⛔ 严禁用于数据中心高并发!"]
    B -- "数据中心 / 云端 GPU 集群<br/>(高并发连续批处理)" --> D{"核心模型与业务负载特征？"}
    D -- "多模型共享中台 (20+ 模型) /<br/>跨 NVIDIA + AMD + TPU 异构底座" --> E["🏆 主力选 vLLM V1<br/>• Day-0 模型支持最广 · 运维最稳"]
    D -- "DeepSeek 671B 等大 MoE (EP=64) /<br/>高频多轮 Agent + 严格 JSON Schema" --> F["🏆 主力选 SGLang<br/>• RadixAttention + DeepEP + 双批次重叠"]
    D -- "单一固定超大流量模型 /<br/>Blackwell B200 追求极限 NVFP4 低延迟" --> G["🏆 主力选 TensorRT-LLM<br/>• AOT C++ 算子极致融合 (-12% TPOT)"]
    E & F & G --> H["🌐 集群规模 >= 16 卡时: 顶层统一套上 NVIDIA Dynamo 控制面<br/>(KV-Aware Router + NIXL RDMA PD 分离 + 跨机 KV 缓存池)"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（五大推理技术栈全方位对比）

| 评估维度 | `Ollama / llama.cpp` | `vLLM V1` | `SGLang` | `TensorRT-LLM` | `NVIDIA Dynamo` + (`SGLang / vLLM / TRT-LLM`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **核心定位** | 端侧/本地轻量运行时 (`GGUF`) | 通用云原生推理引擎标杆 | Agent Radix 缓存与大 MoE 引擎 | AOT 编译级硬件极限加速引擎 | **集群级 PD 分离与 KV 感知编排控制面** |
| **32x H200 达标 `Goodput/GPU`** | `290 tok/s` ✕（数据中心反模式） | `1,960 tok/s` ✓ | `2,280 tok/s` ✓ | `2,120 tok/s` ✓ | **`2,490 tok/s` ✓（集群全局最优）** |
| **多轮 Agent 前缀缓存命中率** | `15.0%` | `76.4%` | `85.2%` (`RadixAttention`) | `74.0%` | **`92.8%` ✓（全局 KV-Aware 路由）** |
| **新模型 Day-0 接入与易用性** | 极快（社区 GGUF） | **极快（HuggingFace 直载）** | **极快（HuggingFace 直载）** | 较慢（需编译 `.engine`） | 继承底层引擎（支持混挂三者） |
| **跨芯片生态（AMD / TPU）** | CPU / Apple M 系 / Vulkan | **最强（NVIDIA + AMD + TPU）** | 强（NVIDIA + AMD） | 仅限 NVIDIA GPU | 控制面开源通用，`NIXL` 深度优化 NVIDIA |
| **每百万达标 Token 成本** | `$2.95 / 1M` ✕ | `$0.44 / 1M` | `$0.38 / 1M` | `$0.41 / 1M` | **`$0.34 / 1M` ✓（降低 `88.5%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **严禁把 `Ollama` 直接推上企业高并发生产环境**：超过 `8` 并发必须切换至 `vLLM V1` 或 `SGLang`。
- [ ] **常规 Dense/多模态多模型平台默认选 `vLLM V1`；`DeepSeek-V3/R1` 大 MoE 与重度多轮 Agent 优先选 `SGLang`**。
- [ ] **多副本（`Replicas >= 4`）部署时禁止使用普通 Nginx 轮询**：必须前置 `Dynamo KV-Aware Router` 或 `vLLM Production Stack / Mooncake` 前缀感知路由。
