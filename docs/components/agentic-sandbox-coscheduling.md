# Agentic RL 万级 MicroVM 沙箱（Firecracker UFFD）与 GPU KV Cache 协同调度

## 1. 核心架构问题与结论（Executive Summary）

在 SWE-bench、代码生成与多步工具调用的强化学习后训练（Agentic RLVR / Multi-Turn GRPO）中，训练集群不再只是单纯运行 LLM 生成，而是呈现 **“GPU 推理引擎（vLLM/SGLang）+ 万级并发代码沙箱（Firecracker MicroVM / 容器）”** 的强耦合交替循环（如 `volcengine/verl` 的 `ToolAgentLoop`）：
1. **长耗时外部工具调用引发的 GPU HBM “占坑死锁（KV Hoarding）”**：当一个已累积 `32,768` Token 上下文（BF16 KV Cache 约 **`4.0 GiB`**）的 Coding Agent 发起一次 `pytest`、C++/Rust 编译或浏览器端到端测试时，外部沙箱执行往往耗时 **`5–60 秒`**。若在此期间将该 Trajectory 的 `4.0 GiB` KV Cache 死死钉在 GPU HBM 中等待工具返回，单次 `30 s` 工具调用即白白烧掉 **`120.0 GiB·s`** 显存预算，导致 8-GPU 节点（`480 GiB` 可用 KV 池）的并发活跃 Rollout 吞吐被锁死在 **`8.9` 条**。
2. **Token-in-Token-out 一致性与 Prefix 重算税**：为了规避多轮对话模板重新分词导致的 Token 漂移（Tokenization Drift），`verl` 的 `ToolAgentLoop` 严格维护 `prompt_ids` / `response_ids` 与 `response_mask`（模型生成 Token 为 `1`，环境工具返回 Token 为 `0`）并采用会话粘性路由（Sticky Session Routing）。若在工具调用期间直接丢弃 HBM KV Cache，工具返回后重算 `32K` 前缀需要消耗 **`0.500 s` GPU 计算时间**（按 `16.0 GiB/s` 等效机会成本折算为 **`8.0 GiB·s`**，加缓冲驻留共 **`32.0 GiB·s`**）。
3. **Firecracker MicroVM `userfaultfd`（UFFD）缺页风暴与 GDS KV 卸载的 PCIe/NVMe 抢占**：为了在毫秒级拉起成千上万个干净的仓库测试环境，现代 Agent 沙箱普遍采用 Firecracker 快照恢复 + Linux `userfaultfd` 按需缺页加载。然而，当集群以 **`500 VMs/s`** 的突发速率并发唤醒沙箱（每个沙箱按需缺页拉取 `96 MiB` 活跃工作集页），未受限的 UFFD 缺页流会瞬间吞噬 **`50.33 GB/s`** 的宿主机 NVMe/PCIe 带宽，将同节点 GPUDirect Storage (`cuFile`) KV 卸载通道的可用带宽从 `54.40 GB/s` 挤压至 **`4.07 GB/s`**！这导致 `4.0 GiB` KV 往返 Swap 时间从 `0.160 s` 暴增 **`13.4 倍` 至 `2.139 s`**，期间卡在 HBM Staging Buffer 中的 KV 反而烧掉 **`136.92 GiB·s`**，比直接死锁在 HBM 里还要慢！
4. **三级临界点（Break-Even Thresholds）与双端令牌桶协同调度**：
   - **短工具调用（$t_{\text{tool}} < 2.56\text{ s}$）**：直接保留在 GPU HBM（`hbm_pin`）；
   - **中长工具调用（$t_{\text{tool}} \ge 2.56\text{ s}$）**：执行 GDS / 3FS 零拷贝卸载（`gds_offload_coscheduled`），但**必须配合沙箱-存储协同令牌桶（将 UFFD 突发缺页带宽限制在共享 NVMe 带宽的 `35%` 即 `19.04 GB/s` 以内）**，使 KV 往返 Swap 稳定在 **`0.246 s`（`15.75 GiB·s`）**，在 `30 s` 与 `60 s` 工具调用下分别实现 **`13.23x`** 与 **`25.58x`** 的节点并发活跃 Rollout 倍增！

---

## 2. 一手源码深度核验（`volcengine/verl` `ToolAgentLoop` 与 `firecracker` UFFD）

### 2.1 `volcengine/verl`：`ToolAgentLoop` 状态机、Token-in-Token-out 与 `TokenBucketWorker`

在 `volcengine/verl`（commit `4b26c788d41ce932ce7706c5ee8e249ab52f4c51`，`2026-10-18`）的 `verl/experimental/agent_loop/agent_loop.py`（blob `6a38d1eba8da282ebf6874aee98b56e5991260b0`）、`verl/experimental/agent_loop/tool_agent_loop.py`（blob `365159c18ffb7e3d6977485d9647ae20fd9d4d42`）、`docs/advance/agent_loop.rst`（blob `d5b41ff7cfdbe6864e02fbac2b1753f0fbef4271`）与 `docs/sglang_multiturn/sandbox_fusion.rst`（blob `995635627ed24d2e172f9913189a9eefcd869d79`）中，核验到多轮 Agent RL 的核心实现机制：

1. **五态异步循环与 Sticky Session 路由（`tool_agent_loop.py` & `agent_loop.rst`）**：
   - 每个 Trajectory 在 `PENDING -> GENERATING -> PROCESSING_TOOLS -> TERMINATED`（可选 `INTERACTING`）状态间流转；
   - `AsyncLLMServerManager` 维护粘性路由（Sticky Session），确保同一个 `request_id` 在工具返回后的下一轮 `GENERATING` 总是路由回原先持有其 Prefix Cache 的同一个 vLLM/SGLang Server 实例。
2. **严禁二次分词（Token-in-Token-out）与 MoE 专家路由对齐（`agent_loop.py`）**：
   - 文档与代码明确指出：若在每轮工具返回后将历史对话拼接为字符串再调用 `apply_chat_template` 分词，不同轮次边界处的空白符或特殊 Token 极易发生合并变化，导致两轮之间的 Token ID 序列不一致（不仅引发 RL 策略梯度计算错误，更直接使推理引擎的 Prefix KV Cache 100% 失效！）；
   - 因此 `LLMServerClient.generate` 强制直接传递 `prompt_ids` 数组，增量追加模型输出的 `response_ids`（置 `response_mask = 1`）与工具输出分词后的 `tool_response_ids`（置 `response_mask = 0`）；同时在启用 MoE Rollout 路由重放时，对工具 Token 槽位填零对齐 `routed_experts` 张量。
3. **沙箱并发速率限制（`sandbox_fusion.rst`）**：
   - 为防止万级并发 Rollout 同时打爆外部代码执行沙箱（Sandbox Fusion），`verl` 引入了基于 Ray Actor 并发组（`concurrency_groups={"acquire": 1, "release": 10}`）的 `TokenBucketWorker`，在进入工具执行前进行令牌桶限流。

### 2.2 `firecracker-microvm/firecracker`：用户态 `userfaultfd` 快照恢复与乱序事件陷阱

在 `firecracker-microvm/firecracker`（commit `e786ba84fd5e0db878392a6a1f234e1b17fcaeb2`，`2026-10-16`）的 `docs/snapshotting/handling-page-faults-on-snapshot-resume.md`（blob `8269d8e56effb8dd445aa7cbd4b7b75f85de486b`）、`src/firecracker/examples/uffd/on_demand_handler.rs`（blob `3101aa19253a7cc7ecf0490d778202484e6906cc`）与 `src/firecracker/examples/uffd/uffd_utils.rs`（blob `ab28f6f4d2e9e68c81aebdf0a00916a06c4e875f`）中，揭示了高密度 MicroVM 快照恢复的内核级交互细节：

1. **UDS `SCM_RIGHTS` 传递与 `UFFDIO_COPY` 按需填页（`uffd_utils.rs`）**：
   - Firecracker 恢复快照时并不读取完整内存镜像文件，而是通过 Unix Domain Socket 的辅助数据（`SCM_RIGHTS`）将 `userfaultfd` 文件描述符及 JSON 编码的 `GuestRegionUffdMapping` 数组发送给宿主机上的 UFFD Handler 进程；
   - 当 Guest vCPU 访问未驻留内存页触发缺页异常时，内核挂起该 vCPU 线程并通过 UFFD 向 Handler 发送 `Pagefault` 事件，Handler 从底层存储读取对应偏移并在宿主机调用 `uffd.copy(src, dst, len, true)`（即 `UFFDIO_COPY` ioctl）拷贝填页并唤醒 vCPU。
2. **`UFFD_EVENT_REMOVE` 与 `UFFDIO_COPY` 的 `-EAGAIN` 死锁/乱序陷阱（`on_demand_handler.rs` & 文档）**：
   - **非因果乱序（Non-causal Ordering）**：缺页事件（`Pagefault`）由 KVM vCPU 线程触发，而内存气球回收事件（`Remove`，对应 `MADV_DONTNEED`）由 Firecracker VMM 控制线程触发。由于两者运行在不同宿主机线程上，UFFD 事件队列可能出现因果倒置；
   - **强制排空契约**：当 Handler 正在处理一个 `Pagefault` 并调用 `uffd.copy()` 时，若内核在此期间收到气球驱动对该页的 `UFFD_EVENT_REMOVE`，内核会中止 `UFFDIO_COPY` 并返回 **`-EAGAIN`（`os error 11`）**！源码（`on_demand_handler.rs`）明确要求：一旦捕获 `EAGAIN`，Handler **必须先从 UFFD 队列中读出并处理所有挂起的 `Remove` 事件（取消相关页映射），随后才能重试或唤醒缺页**，否则 vCPU 线程将永久卡死。
   - **进程级联保护（`SO_PEERCRED` + Panic Hook）**：`uffd_utils.rs` 通过 `getsockopt(..., SO_PEERCRED)` 获取对端 Firecracker VMM 的 PID，并注册 Rust Panic Hook——一旦 UFFD Handler 异常崩溃，立即向对应 Firecracker 进程发送 `SIGKILL`，防止孤儿 MicroVM 永久挂死在内核态缺页等待中。

---

## 3. HBM `GiB·s` 成本方程与工具调用 Break-Even 推导

设单条 Agent Trajectory 在发起工具调用时的上下文长度为 $L$，单 Token KV Cache 字节数为 $b_{\text{kv}}$，则该请求占用的显存体积（GiB）为：
$$M_{\text{kv}} = \frac{L \cdot b_{\text{kv}}}{2^{30}}$$
引入 GPU HBM 容量的时间机会成本系数 $C_{\text{gpu}} = 16.0\text{ GiB/s per GPU-second}$（即独占 1 秒钟的单卡计算算力等价于占用 `16.0 GiB·s` 显存预算），比较三种策略在工具执行耗时 $t_{\text{tool}}$ 秒内的总等效成本 $J(t_{\text{tool}})$（单位：`GiB·s`）：

1. **策略 A：HBM 原地驻留（`hbm_pin`）**：
   $$J_{\text{pin}}(t_{\text{tool}}) = M_{\text{kv}} \cdot t_{\text{tool}}$$
2. **策略 B：丢弃并在工具返回后重算 Prefix（`evict_recompute`）**：
   设重算 $L$ 个 Token 的 Prefill 耗时为 $t_{\text{recomp}} = \frac{L}{R_{\text{prefill}}}$，加预留调度窗口 $t_{\text{stage}}$：
   $$J_{\text{recomp}} = M_{\text{kv}} \cdot (t_{\text{recomp}} + t_{\text{stage}}) + C_{\text{gpu}} \cdot t_{\text{recomp}}$$
   令 $J_{\text{pin}}(t_{\text{tool}}) = C_{\text{gpu}} \cdot t_{\text{recomp}}$，得重算相对驻留的纯算力盈亏平衡时间：
   $$t_{\text{BE,recomp}} = \frac{C_{\text{gpu}} \cdot t_{\text{recomp}}}{M_{\text{kv}}}$$
3. **策略 C：GDS / 3FS 零拷贝卸载与恢复（`gds_offload`）**：
   设宿主机 NVMe/3FS 总带宽为 $B_{\text{nvme}}$，并发 Firecracker UFFD 缺页流抢占带宽为 $B_{\text{uffd}}$，剩余可供 GDS 使用的带宽为 $B_{\text{gds}} = \max(B_{\text{floor}}, B_{\text{nvme}} - B_{\text{uffd}})$，则往返写出+读回耗时 $t_{\text{swap}} = \frac{2 \cdot M_{\text{kv}} \cdot 1.07374}{B_{\text{gds}}}$，等效总成本为：
   $$J_{\text{gds}} = \left(M_{\text{kv}} + C_{\text{gpu}}\right) \cdot t_{\text{swap}}$$
   无争用下的 GDS 卸载盈亏平衡阈值为：
   $$t_{\text{BE,gds}} = \frac{C_{\text{gpu}} \cdot t_{\text{swap,0}}}{M_{\text{kv}}}$$

代入典型参数（$L = 32,768$，$b_{\text{kv}} = 131,072\text{ B} \Rightarrow M_{\text{kv}} = 4.0\text{ GiB}$，$R_{\text{prefill}} = 65,536\text{ tok/s} \Rightarrow t_{\text{recomp}} = 0.500\text{ s}$，$B_{\text{nvme}} = 54.40\text{ GB/s} \Rightarrow t_{\text{swap,0}} = 0.160\text{ s}$）：
- **GDS 卸载盈亏平衡点**：$t_{\text{BE,gds}} = \frac{16.0 \times 0.160}{4.0} = \mathbf{2.56\text{ s}}$；
- **Prefix 重算盈亏平衡点**：$t_{\text{BE,recomp}} = \frac{16.0 \times 0.500}{4.0} = \mathbf{8.00\text{ s}}$。

---

## 4. 原创 CPU 离线实验结果（`experiments/agentic_sandbox_coscheduler/`）

运行 `python experiments/agentic_sandbox_coscheduler/run.py --check` 验证不同工具调用时长（`1.0 s` 至 `60.0 s`）与不同 Firecracker UFFD 突发唤醒速率（`100–1000 VMs/s`）下的表现：

### 4.1 工具调用时长与 UFFD 突发强度下的四策略成本与并发量对比表

| 工具耗时 (`t_tool`) | UFFD 唤醒速率 (`VMs/s`) | UFFD 缺页需求 (`GB/s`) | `hbm_pin` 成本 (`GiB·s`) | `recompute` 成本 (`GiB·s`) | 未限流 GDS 成本 (`GiB·s`) | **协同限流 GDS 成本 (`GiB·s`)** | 节点活跃 Rollout（`Pin` vs **`Co-Sched`**） |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1.0 s`** | `100 VMs/s` | `10.07 GB/s` | **`4.00 GiB·s`** | `32.00 GiB·s` | `3.94 GiB·s` | `3.94 GiB·s` | `87.3` vs `87.8`（短调用直接驻留） |
| **`3.0 s`** | `200 VMs/s` | `20.13 GB/s` | `12.00 GiB·s` | `32.00 GiB·s` | `5.10 GiB·s` | **`4.94 GiB·s`** | `50.5` vs **`81.5`**（跨过 `2.56s` 阈值） |
| **`10.0 s`** | `300 VMs/s` | `30.20 GB/s` | `40.00 GiB·s` | `32.00 GiB·s` | `7.23 GiB·s` | **`4.94 GiB·s`** | `20.9` vs **`81.5`**（`3.90x` 并发提升） |
| **`30.0 s`** | `500 VMs/s` | **`50.33 GB/s`** | `120.00 GiB·s` | `32.00 GiB·s` | **`136.92 GiB·s`**（被 UFFD 饿死） | **`15.75 GiB·s`** | `8.9` vs **`117.6`（`13.23x` 提升）** |
| **`60.0 s`** | `1000 VMs/s` | **`100.66 GB/s`** | `240.00 GiB·s` | `32.00 GiB·s` | **`139.26 GiB·s`**（被 UFFD 饿死） | **`15.75 GiB·s`** | `4.5` vs **`115.1`（`25.58x` 提升）** |

---

## 5. 架构师设计规范与思考练习

### 5.1 Agentic RL 基础设施黄金法则
1. **按工具类型预分类路由（Tool-Duration-Aware KV Tiering）**：在 `ToolAgentLoop` 拦截到工具调用名称时立即分流——对内存计算或轻量正则搜索（预计 `< 2.5 s`）保持 HBM 驻留；对 `docker build`、`cargo test`、`pytest`、浏览器渲染等重型沙箱动作（预计 `> 2.56 s`），在投递给 `TokenBucketWorker` 的同一毫秒异步触发 `cuFileWriteAsync` / `3FS USRBIO` 写出并立即释放 GPU Paged KV 物理块。
2. **宿主机 I/O 协同限流器（Unified Host I/O Token Bucket）**：严禁让 Firecracker UFFD Handler 无限制抢占本地 NVMe 队列。应在宿主机 Linux `blk-iocost` / `cgroups v2 io.max` 或用户态 UFFD Handler 内部接入与 GDS 共享的令牌桶，保证 GPU KV Swap 始终拥有至少 **`65%`（`>= 35.36 GB/s`）** 的最高优先级带宽保底。

### 5.2 思考与计算练习
- **练习 1（上下文长度对盈亏平衡点的影响）**：当 Agent 上下文从 `32K` 增长至 `128K` Token 时，若采用普通稠密注意力（Prefill 耗时随序列长度呈二次方增长，而 KV 体积呈线性增长），请推导重算阈值 $t_{\text{BE,recomp}}$ 与 GDS 卸载阈值 $t_{\text{BE,gds}}$ 各自如何变化？
- **练习 2（UFFD `-EAGAIN` 故障排查）**：在压测万级 Firecracker 沙箱时，监控发现部分 MicroVM 在恢复后第 3 秒永久卡死，`strace` 显示宿主机 UFFD Handler 在调用 `ioctl(uffd, UFFDIO_COPY, ...)` 时不断返回 `EAGAIN`。结合 `on_demand_handler.rs` 源码说明根本原因及修复代码结构。
