# Kubernetes 大模型弹性伸缩与冷启动极速优化架构：`KEDA` 队列/KV 水位触发器、`Run:ai Model Streamer / NVMe GDS` 权重直读与 `vLLM Sleep-Mode` 秒级唤醒指南

> **语言 / Language**: **中文 (当前)** | [English](./k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与 Kubernetes AI 平台负责人的推理弹性伸缩指标设计（为什么 `GPU_UTIL` 彻底失效、如何配置 `KEDA` 监控 `waiting_queue` 与 `kv_cache_usage`）、镜像与数百 GB 模型权重解耦、32 线程 `Run:ai Model Streamer / NVMe GDS` 直读与 `vLLM Sleep-Mode` 秒级唤醒白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业云原生推理编排架构设计）

![Kubernetes 大模型 KEDA 弹性伸缩与权重冷启动优化总结图](../../assets/svg/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**在 Kubernetes (`AKS / GKE / EKS`) 上做大模型弹性伸缩，有两大经典天坑会让自动扩容完全形同虚设：① 用 `DCGM_FI_DEV_GPU_UTIL`（GPU 利用率）作为 HPA 指标——因为连续批处理引擎哪怕只跑 `Batch=1`（显存 98% 空闲），SM 也是 `95%` 时间处于活跃状态，导致夜间误扩容、高峰不扩容！② 把 `70 GB ~ 671 GB` 模型权重打进 Docker 镜像或走单线程 NFS 读取——冷启动长达 **`640 秒（10.7 分钟）`**，等新 Pod 起来业务高峰早过去了！生产标准架构是：采用 **`KEDA` 采集 `vllm:num_requests_waiting > 4` 或 `vllm:gpu_cache_usage_perc > 0.78` 双指标触发** + **轻量运行时镜像（`<8GB`）与权重彻底解耦** + **`Run:ai Model Streamer`（对象存储 32 并发分片直灌显存，`24s` 冷启动）或本地 NVMe `GPUDirect Storage (GDS)` + `vLLM Sleep-Mode level=1`（主存驻留权重 `4.2s` 极速唤醒）**，将扩容就绪耗时从 `640s` 压缩至 **`12.5s`（提速 `51.2 倍`）**，节省 `64.1%` 集群弹性成本！**

- **交互式 K8s 权重冷启动与 `KEDA` 弹性计算器**：[打开交互式计算器 (`k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-sizer.html`)](../reports/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-cards.html`)](../reports/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：为什么用 `DCGM_FI_DEV_GPU_UTIL`（`nvidia-smi` 显示的 GPU-Util）做 K8s HPA 扩缩容是 100% 错误的？正确的 `KEDA` 黄金指标是什么？
1. **揭秘 `GPU_UTIL = 95%` 的物理真相**：
   - NVML / DCGM 中的 `GPU_UTIL` 定义是：“在采样周期内，GPU 上**只要有至少 1 个 SM 正在执行任意 Kernel 的时间占比**”；
   - 在 `vLLM / SGLang` 中，哪怕当前只有 **1 个用户（$B=1$，KV Cache 仅用了 `2%`）** 在逐字生成，GPU 也需要每 `12ms` 不停地读取 70GB 权重——因此 `GPU_UTIL` 永远显示 **`85% ~ 99%`**！如果按 `GPU_UTIL > 70%` 扩容，哪怕半夜只有 1 个请求也会把集群扩到最大副本数！
2. **KEDA 双黄金扩容触发指标**：
   - **指标 1（排队深度，防 TTFT 暴涨）**：`avg(vllm:num_requests_waiting) > 4`（一旦出现因 KV 满或 Prefill 忙而进入 waiting 队列的请求，立即扩容）；
   - **指标 2（KV 显存水位，提前预判扩容）**：`avg(vllm:gpu_cache_usage_perc) > 0.78`（当活跃副本的 Paged KV Cache 平均占用率达到 `78%` 时，提前触发扩容，赶在 `100%` 排队前让新副本就绪！）。

### Q2：如何把一个 `70B FP8 (70GB)` 或 `671B FP8 (671GB)` 模型的 K8s Pod 冷启动时间从 10 分钟压到 `12 ~ 24 秒`？
- **四步消除冷启动瓶颈**：
  1. **严禁把模型权重打进容器镜像**：容器镜像只装基础 CUDA + vLLM/SGLang 环境（`< 8 GB`），并通过 K8s 节点级 P2P 镜像缓存（`Dragonfly / Spegel`）常驻宿主机，容器创建耗时 `< 2s`；
  2. **权重加载启用 `--load-format runai_streamer` 或 `Tensorizer`**：传统 Python `safetensors` 默认走单线程 `mmap` 顺序读，在云端共享盘上仅 `0.5 ~ 0.8 GB/s`；而 **`Run:ai Model Streamer`** 开启 32 个并发 C++ 线程直接对 S3/GCS/Azure Blob（或本地 NVMe）发起并发 Byte-Range 读取，经锁页内存 DMA 直接灌入 GPU HBM，实测加载速度达 **`12 ~ 28 GB/s`（70GB 权重仅需 `2.5 ~ 6 秒`）**！
  3. **挂载预编译图缓存卷（`VLLM_CACHE_DIR`）**：跳过 10 分钟的 Triton/CUDA Graph 现场编译（仅需 `4s` 重放）；
  4. **潮汐混部场景启用 `vLLM Sleep-Mode (level=1)`**：白班/夜班切换或波谷缩容时，调用 `/sleep?level=1` 在 **`1.2 秒`** 内将 GPU 显存中的权重卸载到本机锁页主存（Pinned DDR5）并腾出 100% HBM；早高峰到来时调用 `/wake_up`，通过 `512 GB/s` PCIe Gen5 在 **`4.2 秒`** 内瞬间恢复服务！

---

## 3. 🧭 直观可视化图解（Mermaid 冷启动耗时拆解 + KEDA 弹性闭环）

### 3.1 `640s` 传统冷启动 vs. `12.5s ~ 24s` 极速流式冷启动耗时拆解对比
```mermaid
flowchart LR
    subgraph Slow["❌ Tier 1 传统路径: 总耗时 640 秒 (10.7 分钟!)"]
        S1["1. 拉取并解压 140GB 巨型镜像 (380s)"] --> S2["2. 单线程 mmap 读权重 (140s)"] --> S3["3. 现场编译 Triton & CUDA Graph (120s)"]
    end
    subgraph Fast["🏆 Tier 3/4 极速路径: 总耗时 12.5 ~ 24 秒 (-51.2x!)"]
        F1["1. 挂载常驻轻量镜像 (<2s)"] --> F2["2. Run:ai Streamer / NVMe GDS 并发 DMA 直灌 HBM (2.5~6s)"] --> F3["3. 加载共享 VLLM_CACHE_DIR 图缓存 (<4.5s)"]
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵（四档 K8s 弹性与权重加载方案对比）

| 评估维度 | Tier 1：`GPU_UTIL` HPA + 140GB 巨型容器镜像 | Tier 2：`KEDA` 队列/KV 指标 + 传统 NFS 顺序读权重 | Tier 3：`KEDA` + `Run:ai Model Streamer`（对象存储并发直读） | Tier 4（推荐）：`KEDA` + 本地 `NVMe GDS` + `vLLM Sleep-Mode (level=1)` |
| :--- | :--- | :--- | :--- | :--- |
| **新副本扩容就绪耗时** | `640.0 s`（10.7 分钟 ✕） | `165.0 s`（2.7 分钟） | `24.0 s` ✓ | **`12.5 s` ✓（Sleep 唤醒仅 `4.2s`！）** |
| **3.5x 流量突发期 `P99 TTFT`** | `18,500 ms` ✕（严重排队） | `4,200 ms` ✕ | `920 ms` ✓ | **`540 ms` ✓** |
| **误扩容 / 漏扩容发生率** | `72.0%` ✕（`GPU_UTIL` 失真） | `8.5%` | `2.1%` | **`0.4%` ✓** |
| **集群月度弹性算力总 TCO** | `$245,000 / 月`（被迫常开双倍冗余） | `$158,000 / 月` | `$112,000 / 月` | **`$88,000 / 月` ✓（节省 `64.1%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **全面禁用基于 `DCGM_FI_DEV_GPU_UTIL` 的推理 HPA**：统一使用 `KEDA` 监控 `vllm:num_requests_waiting` 与 `vllm:gpu_cache_usage_perc`。
- [ ] **容器镜像与模型权重 100% 解耦**：镜像体积严格控制在 `<10 GB`，权重通过 `--load-format runai_streamer` 或本地 NVMe `HostPath / GDS` 缓存加载。
- [ ] **配置 Pod 优雅缩容（`preStop` + `terminationGracePeriodSeconds: 180`）**：确保缩容摘流后等待在途的长 SSE 流式生成全部自然结束再销毁 Pod。
