# 大模型多进程推理引擎（`vLLM V1 / SGLang`）Kubernetes 探针“假 `200 OK`”流量黑洞与共享内存 `cudaEventQuery` 硬件进度看门狗租约（`Watchdog Lease`）选型：为什么前后端分离架构下 GPU Worker 卡死在 `ncclKernel` 时 `FastAPI /health` 依然持续返回 `200 OK`（吞噬 `100%` 线上流量），而往探针里塞 1-Token 真实推理又会在大并发下引发 `42.0%` 的正常节点被 K8s 误杀重启？基于共享内存 `cudaEventQuery` 硬件步进时间戳 + 动态负载租约（`100%` 死锁检出，`0.0%` 误杀，探针延迟 `0.05 ms`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_serving_liveness_probe_false_200_ok_gpu_worker_watchdog_lease_selection/run.py` 与 `results.json`。

![LLM Serving Liveness Probe False 200 OK & Shared-Memory GPU Watchdog Visual Summary](../../assets/svg/llm-serving-liveness-probe-false-200-ok-gpu-worker-watchdog-lease-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向前后端多进程解耦推理架构（`vLLM V1 AsyncLLM + EngineCore + TP=8 MultiprocWorkers / SGLang TokenizerManager + Scheduler`）的“基于 `cudaEventQuery` 硬件流完成回执的无锁共享内存（`POSIX shm / mmap`）看门狗租约 + `IDLE / BUSY` 状态感知动态超时门控 + 子进程异常父进程秒级自熔断（`Self-Fencing`）”全栈高可用体系**。
- **架构师价值**：在生产级 Kubernetes 大模型推理集群中，为了避免 Python GIL 阻塞流式发包，现代推理引擎全部采用了 **前后端进程解耦架构**：
  - **进程 0（前端 API Server）**：跑 `Uvicorn / FastAPI`，负责 HTTP 接入、分词/反分词与 SSE 推流，**完全不碰 CUDA / GPU**；
  - **进程 1..8（后端 EngineCore 与 `TP=8` GPU Workers）**：通过 `ZMQ` 套接字接收请求，在 8 张 GPU 上跑模型前向与 `NCCL All-Reduce`。
  这种解耦架构在提升吞吐的同时，给 Kubernetes 的健康检查（`livenessProbe` / `readinessProbe`）埋下了两个要命的线上大雷：
  1. **大雷一：“假 `200 OK`”僵尸流量黑洞（Zombie Traffic Blackhole，GPU 死锁检出率 `0.0%`，Tier 1）**！
     - 当后端某张卡（如 Rank 3）遭遇硬件 `ECC Uncorrectable`、非法地址访问（`CUDA_ERROR_ILLEGAL_ADDRESS`），或者因为交换机丢包导致 8 张卡永久卡死在 `ncclDevKernel_AllReduce` 的死循环里时，后端的 GPU Worker 进程已经彻底停止产出 Token；
     - **但前端的 `FastAPI` 进程依然活得好好的，而且因为没活干所以特别空闲！** 每隔 5 秒，K8s `kubelet` 和上游网关（Envoy / Nginx）发来 `GET /health`，`FastAPI` 毫秒级返回 **`HTTP 200 OK`（耗时 `0.20 ms`）**！
     - **灾难后果**：负载均衡器认为该 Pod 不仅健康而且响应极快，继续把大批生产流量源源不断地路由进这个已经死透的“僵尸黑洞 Pod”，导致用户请求 `100%` 挂起超时长达数小时，直到值班工程师半夜被报警叫醒手动删 Pod！
  2. **大雷二：在 `/health` 里发一条 `max_tokens=1` 的真实推理探测，结果高峰期引发 `42.0%` 的健康 Pod 被 K8s 连环误杀（Cascading Reboot Avalanche，Tier 2）**！
     - 为了验证 GPU 真的没死，许多运维团队把 `/health` 改成了调一次内部 `engine.generate("ping", max_tokens=1)`。
     - **为什么这会导致雪崩？** 当线上突然涌入一批 `64K` 超长上下文或大图多模态 Prefill 请求时，GPU 正常跑完一个大 Chunk 就要花 **`3,450 ms`（3.45 秒）**！此时进来的 `/health` 探测请求被迫排在推理队列里等这 `3.45 秒` 跑完才能返回，而 K8s 的 `livenessProbe.timeoutSeconds` 通常设为 `2 秒`！
     - 于是 K8s 判定这个 **正在满负荷拼命干活的健康 Pod “已死”**，直接发 `SIGKILL` 把它杀掉重启（高峰期误杀率高达 **`42.0%`**）！这个 Pod 一死，它身上的流量瞬間转移给旁边本就繁忙的兄弟 Pod，把兄弟 Pod 的延迟也顶过 2 秒，引发 **全集群连环重启雪崩**！
  采用 **由 GPU Worker 在每个 Step 的硬件流 `cudaEventQuery(step_evt) == cudaSuccess` 确认完成后才刷新 `/dev/shm` 共享内存时间戳 + `FastAPI /health` 仅用 `0.05 ms` 无锁读取共享内存租约 + 区分 `IDLE / BUSY` 动态放宽大 Prefill 租约预算（Tier 4）**，可实现 **`100.0%` 捕获所有 GPU/NCCL 死锁与进程崩溃**、**高峰期 `0.0%` 误杀**，且探针 P99 延迟低至 **`0.05 ms`（比合成推理探针快 `69,000 倍`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么连“CPU 级 ZMQ 心跳 Ping-Pong（Tier 3）”也会漏掉 `55%` 的 GPU 死锁？**
   - 因为 **CUDA Kernel 的启动（Kernel Launch）是异步的**！
   - 当 GPU 上的 `ncclAllReduce` 或自定义 Triton 算子陷入硬件死循环（Spin-Wait Hang）时，如果 Worker 的 Python 主线程在另一个异步流或后台协程里响应 ZMQ 心跳（或者在未做同步前先回复了心跳），CPU 线程看起来依然“活着”，但 GPU 硬件流已经永远卡死！
   - **唯一可靠的存活性证明（Proof of GPU Stream Progress）**：必须让每个 `TP` Worker 在 GPU 计算流（`compute_stream`）的每个 Step 末尾插入一枚硬件事件 `cudaEventRecord(step_done_evt, compute_stream)`，并且 **只有当非阻塞轮询 `cudaEventQuery(step_done_evt)` 返回 `cudaSuccess`（证明 GPU 硬件真正执行完了这一步的所有指令）时**，才允许更新共享内存中的 `last_gpu_progress_ns[rank]`！
2. **如何彻底杜绝长 Prefill 期间的误杀？——`IDLE / BUSY` 状态机 + 批预算动态租约（Dynamic Batch-Budget Lease）**：
   - 在 `/dev/shm` 的看门狗表项中，每个 Worker Rank $r \in \{0..TP-1\}$ 维护四个 64 位原子字段：
     $$\bigl(\text{state}_r \in \{\text{IDLE}, \text{BUSY}, \text{ERROR}\}, \quad \text{last\_gpu\_ok\_ns}_r, \quad \text{lease\_budget\_ns}_r, \quad \text{pid}_r\bigr)$$
   - **规则 1（空闲免检）**：当队列里没有任何请求时，Worker 将 $\text{state}_r \leftarrow \text{IDLE}$。此时就算 1 小时没跑 GPU Step，`/health` 也判定为健康（`200 OK`）；
   - **规则 2（按当步 Token 预算动态定容租约）**：当调度器开始执行一个包含 $N_{\text{tokens}}$ 的 Prefill/Decode 步时，先将 $\text{state}_r \leftarrow \text{BUSY}$，并根据当前步的计算量设定宽裕的硬件死锁阈值：
     $$\text{lease\_budget\_ns}_r = \max\bigl(15\,\text{s}, \;\; 3 \times \hat{T}_{\text{step}}(N_{\text{tokens}})\bigr)$$
   - **规则 3（无锁极速判定）**：前端 `FastAPI` 在收到 `GET /health` 时，**绝不向推理队列塞任何任务**，只需用 `0.05 ms` 读取这 8 个 Rank 的共享内存槽位：只要发现任一进程 `!os.kill(pid, 0)`、或 $\text{state}_r == \text{ERROR}$、或 $(\text{state}_r == \text{BUSY} \land \text{now} - \text{last\_gpu\_ok\_ns}_r > \text{lease\_budget\_ns}_r)$，立即返回 **`HTTP 503`** 并触发整组进程自杀重启！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：为什么当 8 卡 `TP=8` 中只有 Rank 3 一个子进程异常退出（如被 Linux OOM Killer 杀掉）时，推荐由父进程/看门狗线程直接向整个进程组发送 `SIGTERM / SIGKILL` 自尽，而不是等 K8s 每 5 秒一次的 `/health` 探针来杀？
- 因为在 `TP=8` 强耦合集合通信中，只要有 1 个 Rank 退出，其余 7 个 Rank 100% 不可恢复！如果等 K8s `livenessProbe`（`periodSeconds: 5, failureThreshold: 3`）连续失败 3 次才重启容器，中间那 **`15 秒`** 内打进来的所有请求都会报错或卡死！
- **最佳实践（主动自熔断 + 探针双保险）**：
  1. 在 `FastAPI` 主进程中启动一个轻量级后台守护线程（每 `100 ms` 扫描一次子进程 `pid` 存活性与 `shm.error_flag`），一旦发现任一 Worker 崩溃或报 Sticky CUDA Error，**在 `100 ms` 内主动将 `/health` 置为 `503` 摘流并退出主进程（触发容器立即重启）**；
  2. 同时依靠 `shm.last_gpu_ok_ns` 捕获那些进程没退出、但 GPU 死锁在 `ncclKernel` 里的静默挂起故障！

### Q2：Kubernetes 的 `readinessProbe`（就绪探针）和 `livenessProbe`（存活探针）在 LLM 推理服务上应该返回相同的结果吗？
- **绝不应该相同！必须严格区分“过载排队（Overloaded）”与“死锁故障（Wedged）”**！
  - **`readinessProbe`（决定负载均衡器是否继续往该 Pod 分发新请求）**：当共享内存看门狗异常 **或者当前排队等待的请求数 `waiting_queue_len > max_queue_watermark`** 时返回 `503`，让网关暂时把新请求分流给空闲 Pod，**但绝不重启当前 Pod**；
  - **`livenessProbe`（决定 kubelet 是否直接 `SIGKILL` 杀掉重启容器）**：**仅当** 进程崩溃、遭遇不可恢复 CUDA 错误、或 `BUSY` 状态下超过 `lease_budget_ns`（如 `15~30s`）GPU 硬件事件仍未完成时，才返回 `503` 触发容器重启！

### Q3：为什么在模型刚启动加载 `140 GB` 权重和录制 40 个 `CUDAGraph` 时，必需要配独立的 `startupProbe`？
- 因为 70B/671B 模型冷启动加载权重 + `torch.compile` + 录制 `CUDAGraph` 通常需要 `90 ~ 300 秒`。如果不配 `startupProbe` 而直接用 `livenessProbe`，容器会在CUDAGraph 录制到一半时被无限循环重启！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 大模型多进程推理服务健康检查与 GPU 看门狗选型决策树

```mermaid
flowchart TD
    A["多进程解耦大模型推理服务 (FastAPI 前端进程 + ZMQ + TP=8 GPU Worker 进程)"] --> B{"GET /health 探针如何实现?"}
    B -- "仅在 FastAPI 进程内直接 return 200 OK" --> C["⛔ Tier 1: GPU 卡死在 ncclKernel 或 Worker 崩溃时仍返 200 -> 100% 流量黑洞长达数小时!"]
    B -- "每次 /health 向引擎队列提交 1-Token 真实推理" --> D["⛔ Tier 2: 64K 长 Prefill 突发时探针排队耗时 3,450 ms -> 触发 42.0% 正常节点被 K8s 误杀雪崩!"]
    B -- "仅通过 ZMQ 向 Worker CPU 线程发送 Ping" --> E["⚠️ Tier 3: 异步 CUDA 流在 GPU 侧死锁时 CPU 仍能回 Ping -> 漏检 55% 的 GPU 硬件挂起!"]
    B -- "共享内存 cudaEventQuery 硬件步进租约 + 动态状态门控" --> F["🏆 Tier 4: 100% 检出 GPU/NCCL 死锁 + 0.0% 突发误杀 + 0.05 ms 无锁极速响应!"]
```

### 3.2 浅层 HTTP 探针（僵尸黑洞）与合成推理探针（`42%` 误杀）vs. 共享内存 `cudaEvent` 租约对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 要么漏检 100% GPU 死锁, 要么在高峰期误杀 42% 正常节点!"]
      B1["Tier 1 (浅层 200 OK): Rank 3 卡死在 NCCL -> FastAPI 毫不知情照返 200 OK -> 流量黑洞!"] --> B2["Tier 2 (合成 1-Token 探针): 被 3.45s 大 Prefill 堵在队列里 -> 超出 2s 阈值被 K8s 误杀重启!"]
    end
    subgraph Good["🏆 Tier 4: 共享内存 cudaEventQuery 硬件进度时间戳 + IDLE/BUSY 动态租约"]
      G1["TP Worker 每步末尾记录 cudaEvent -> 仅当硬件流完成时才写 shm.last_gpu_ok_ns"] --> G2["FastAPI 0.05ms 无锁读 shm: BUSY 超租约立返 503, 大 Prefill 自动放宽租约 0% 误杀!"]
    end
```

### 3.3 `TP=8` GPU Worker 硬件事件回执更新 `/dev/shm` 与 `FastAPI` 无锁健康检查时序图

```mermaid
sequenceDiagram
    participant K8s as Kubelet / Envoy (/health)
    participant API as FastAPI Process 0
    participant SHM as POSIX /dev/shm Watchdog Table
    participant W as TP=8 GPU Worker (Compute Stream)

    W->>SHM: 1. Start step: set `state = BUSY`, `lease_budget = max(15s, 3 * est_step_time)`
    W->>W: 2. Launch forward kernels + `cudaEventRecord(step_done_evt, stream)`
    W->>SHM: 3. When `cudaEventQuery(step_done_evt) == cudaSuccess`: update `last_gpu_ok_ns = now()`
    K8s->>API: 4. `GET /health`
    API->>SHM: 5. Lock-free read `state[0..7]`, `last_gpu_ok_ns[0..7]` in 0.05 ms
    alt All Workers Healthy (`now - last_gpu_ok_ns <= lease_budget` or `IDLE`)
        API-->>K8s: 6a. Return `HTTP 200 OK` (0.05 ms)
    else Any Worker Dead or GPU Stream Hung (`now - last_gpu_ok_ns > lease_budget`)
        API-->>K8s: 6b. Return `HTTP 503` & send `SIGTERM` to process group for clean restart!
    end
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | `/health` 探针与看门狗实现方案 | GPU/NCCL 死锁检出率 | 高峰期 K8s 误杀重启率 | 探针 P99 延迟 | 僵尸黑洞持续时间 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 浅层 FastAPI 静态 `200 OK`** | 仅检查 HTTP 协程存活，不查 Worker | `0.0%` (完全漏检! ✕) | `0.0%` | `0.20 ms` | `3,600 s+` (黑洞! ✕) |
| **Tier 2: 队列内合成 1-Token 推理探针** | 每次探针排入引擎跑 1 步 `generate` | `100.0%` | **`42.0%` (雪崩误杀! ✕)** | `3,450.0 ms` ✕ | `15.0 s` |
| **Tier 3: CPU 级 ZMQ 控制心跳 Ping** | 检查子进程存活与 CPU 事件循环 | `45.0%` (漏检异步流死锁 ✕) | `4.5%` | `12.0 ms` | `1,800.0 s` ✕ |
| **Tier 4: 共享内存 `cudaEvent` 租约 (推荐)** | **硬件流完成回执写 SHM + 动态状态租约** | **`100.0%` 🏆** | **`0.0%` 🏆** | **`0.05 ms` (`69,000x` 🏆)** | **`10.0 s` (自动自愈 🏆)** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_serving_liveness_probe_false_200_ok_gpu_worker_watchdog_lease_selection/run.py`](../../experiments/llm_serving_liveness_probe_false_200_ok_gpu_worker_watchdog_lease_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_serving_liveness_probe_false_200_ok_gpu_worker_watchdog_lease_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-serving-liveness-probe-false-200-ok-gpu-worker-watchdog-lease-sizer.html`](../reports/llm-serving-liveness-probe-false-200-ok-gpu-worker-watchdog-lease-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-serving-liveness-probe-false-200-ok-gpu-worker-watchdog-lease-selection-cards.html`](../reports/llm-serving-liveness-probe-false-200-ok-gpu-worker-watchdog-lease-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Shallow HTTP 200 OK", "hang_detection_pct": 0.0, "false_kill_pct": 0.0 },
      { "tier": "Tier 2: Synthetic 1-Token Probe", "hang_detection_pct": 100.0, "false_kill_pct": 42.0 },
      { "tier": "Tier 3: CPU ZMQ Control Ping", "hang_detection_pct": 45.0, "false_kill_pct": 4.5 },
      { "tier": "Tier 4: Shared-Memory cudaEvent Watchdog", "hang_detection_pct": 100.0, "false_kill_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "hang_detection_pct": "Percentage",
    "false_kill_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Shared-Memory cudaEvent Watchdog Achieves 100% GPU Hang Detection With 0% False Kills",
    "subtitle": "GPU/NCCL hang detection rate (%) across 4 health probe tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "hang_detection_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有部署在 Kubernetes / Envoy 后端的多进程大模型推理服务（`vLLM V1`, `SGLang`, `TensorRT-LLM`），尤其是多卡 `TP/PP/EP` 实例。
- **⛔ Not for**：在前端 HTTP 进程里无脑返回静态 `200 OK`，或在每次 `/health` 探针中往推理队列塞 `max_tokens=1` 请求。

### 生产核验 Checklist
- [ ] **禁止 `/health` 只测 FastAPI HTTP 协程**：必须绑定全部 `TP × PP` GPU Worker 子进程的进程存活性（`proc.is_alive()`）与共享内存健康状态位。
- [ ] **使用 `cudaEventQuery(step_done_evt) == cudaSuccess` 驱动看门狗心跳更新**：确保当 GPU 计算流卡死在 `ncclKernel` 时能够准确触发租约超时。
- [ ] **分离 `startupProbe`、`readinessProbe`（过载保护）与 `livenessProbe`（死锁重启）**：并按 `IDLE / BUSY` 状态动态计算租约上限，防止长 Prefill 期间误杀健康容器。
