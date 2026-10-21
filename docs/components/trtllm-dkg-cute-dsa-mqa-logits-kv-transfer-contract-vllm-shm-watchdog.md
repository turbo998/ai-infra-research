# TensorRT-LLM CuTe DSL 非整除 `next_n` DSA MQA Logits 切分、KV 传输 Backend 契约与 MPI 停机确认及 vLLM WatchDog 剖析

> **关联上游提交（2026-10-21）**：
> - `NVIDIA/TensorRT-LLM` PR `#19976` (`033f2ea0205341b2f3ad55bbeb5593427030ec78`): `[feat] Default-enable CuTe DSL paged MQA logits on SM100f and support uneven next_n split (#19976)`
> - `NVIDIA/TensorRT-LLM` PR `#20117` (`a9000fe10ecd62973dcf3670573419299b390acb`): `[Disagg] Define the formal KV cache transfer Backend contract (#20117)`
> - `NVIDIA/TensorRT-LLM` PR `#20608` (`16d26df5a7675b0aa13a213305c5dd59e504e9d5`): `[Fix] Reliability fix for RemoteMpiCommSessionServer shutdown (#20608)`
> - `vllm-project/vllm` PR `#55700` (`1c879919786497217b3b5fb0258d1d72f2442cb0`): `[Misc] Add process-wide ThreadedWatchDog for diagnosing worker/engine hangs (#55700)`

---

## 1. 核心架构问题与背景

在大规模投机解码（MTP）、PD 分离 KV 传输以及多进程 Worker 协同调度中，推理引擎面临三类关键工程挑战：

1. **质数 / 非整除 `next_n` 投机窗口下 DSA MQA Logits 的 KV 重复读取**：DeepSeek Sparse Attention (DSA) 的 CuTe DSL 分页 MQA Logits 内核原生支持的 `kernel_atoms = (1, 2, 3)`。当投机步数 `next_n` 为不能被 2 或 3 整除的质数（例如 `next_n = 5` 或 `7`）时，原有的调用方拆分（`_pick_dsl_expand`）只能退化为 `atom = 1, expand_factor = next_n`，迫使 GPU 重复扫描完整分页 KV Cache 达 $5\sim 7$ 次。
2. **PD 分离 KV 传输内存复用竞态与 MPI 远端会话关闭丢包**：
   - 在分离式服务中，传输请求取得逻辑结果（`Outcome`）并不意味着底层 RDMA/NVLink 写入器已停止访问未交付槽位（Undelivered Rows）；若在硬件静默（Quiesce）前回收槽位，会导致静默内存踩踏；
   - `RemoteMpiCommSessionServer` 在等待 Worker 线程退出的数秒窗口内停止轮询 ZMQ `PAIR` 控制套接字，此时客户端 `stop_server_main` 采用 `fire-and-forget` 发送的停机信号极易丢失，导致远端 MPI 服务进程挂死。
3. **vLLM Worker 挂死难以定界与空闲等待期间的误报警风暴**：分布式 Worker 死锁时缺乏全线程 Python 栈自动转储机制；而若简单按固定超时轮询，`MessageQueue.acquire_read` / `acquire_write` 在无请求空闲等待时会频繁触发虚假超时转储，或在同一故障上无限刷屏日志。

---

## 2. 数学模型与核心设计机制

```
+--------------------------------------------------------------------------------------------------+
|  TRT-LLM CuTe DSL Uneven next_n (#19976) + KV Contract (#20117/#20608) + vLLM WatchDog (#55700)  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [1] In-Kernel Uneven next_n Split (_pick_dsl_split & _atom_ctx_offsets, PR #19976)              |
|      When atom == 1 and next_n not divisible by {2, 3}:                                          |
|        num_atoms = ceil(next_n / 3),  in_kernel_atom = ceil(next_n / num_atoms)                  |
|        Taken iff waves_split < waves_atom1 and waves_atom1 >= 16                                 |
|        Causal stagger: offset_i = next_n - ((i+1)*base + min(i+1, rem))                          |
|        * next_n=5 -> num_atoms=2, atom=3, offsets=[2, 0] (reads KV 2x instead of 5x: -60.0%)     |
|        * next_n=7 -> num_atoms=3, atom=3, offsets=[4, 2, 0] (reads KV 3x instead of 7x: -57.14%) |
|                                                                                                  |
|  [2] KV Transfer Backend Contract (#20117) & MPI SHUTDOWN_ACK Handshake (#20608)                 |
|      SlotRegion / LayerGroupRows / View / Outcome(delivered_masks, fault) / Attempt.quiesced()   |
|      Undelivered row memory reusable IFF Attempt.quiesced() == True                              |
|      RemoteMpiCommSessionServer polls control socket every 0.1s during teardown & replies ACK    |
|                                                                                                  |
|  [3] vLLM ThreadedWatchDog (#55700)                                                              |
|      `with get_watch_dog().disable():` refreshes last_feed_time during idle shm_broadcast waits  |
|      One-shot dump per stall: triggers iff (now - last_feed > timeout) & (last_log < last_feed)  |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 CuTe DSL 非整除 `next_n` 内核级切分与因果错位偏移（PR `#19976`）

PR `#19976` 在 `sm_100f`（Blackwell 家族）上默认开启 `use_cute_dsl_paged_mqa_logits=True`，并在 `_pick_dsl_split` 中引入内核级非均匀切分：当 `next_n > 1` 且无法被 $\{2, 3\}$ 整除时，令：

$$\text{num\_atoms} = \left\lceil \frac{\text{next\_n}}{\max(\text{kernel\_atoms})} \right\rceil = \left\lceil \frac{\text{next\_n}}{3} \right\rceil, \quad \text{atom} = \left\lceil \frac{\text{next\_n}}{\text{num\_atoms}} \right\rceil$$

设 $\text{tiles} = \lceil \text{max\_ctx} / 256 \rceil$，对比单原子波次 $\text{waves}_{\text{atom1}} = \lceil B \cdot \text{next\_n} \cdot \text{tiles} / \text{num\_sms} \rceil$ 与切分波次 $\text{waves}_{\text{split}} = \lceil B \cdot \text{num\_atoms} \cdot \text{tiles} / \text{num\_sms} \rceil$：仅当 $\text{waves}_{\text{split}} < \text{waves}_{\text{atom1}}$ 且 $\text{waves}_{\text{atom1}} \ge 16$ 时启用内核级切分（避免在 $B=1$ 极小波次下宽 Tile 开销超过收益）。

对于第 $i \in [0, \text{num\_atoms})$ 个原子（前 `rem = next_n % num_atoms` 个原子承载 `base + 1` 个 Token，其余承载 `base = next_n // num_atoms` 个 Token），其因果上下文截断偏移为：

$$\text{offset}_i = \text{next\_n} - \big(i \cdot \text{base} + \min(i, \text{rem}) + \text{base} + \mathbb{I}[i < \text{rem}]\big)$$

例如 `next_n = 7` 时，$\text{num\_atoms} = 3, \text{base} = 2, \text{rem} = 1$，3 个原子分别承载 $(3, 2, 2)$ 个查询步，因果偏移为 **`[4, 2, 0]`**，将分页 KV Cache 读取次数从 7 次降至 3 次（减少 **`57.14%`**）。同时，当启用 `dsl_kernel_split` 时显式门控关闭 `gvr_emit_kwargs`，避免向仅支持单步的 Top-K 闭环状态注入无效中间指标。

### 2.2 分离式 KV 传输 `Backend` 契约与 MPI 会话可靠停机（PR `#20117` / PR `#20608`）

PR `#20117` 将分离式 KV 缓存传输形式化为严格的租约-视图-尝试（Lease-View-Attempt）契约：
- **`SlotRegion` 与 `LayerGroupRows`**：后端仅注册其 `memory_kind` 支持的连续槽位区域 `(address, nbytes)`；每个 `LayerGroupRows` 以只读对齐数组提供 `layer_group_index`、`object_keys`（`uint8 [rows, width]`）与 `addresses`（`int64 [rows]`）。遇到未识别行的提交直接抛出 `InvalidSubmission`（保证无任何远端通知或内存访问逃逸）。
- **`Outcome` 与 `Attempt.quiesced()` 解耦**：`Outcome(delivered_masks, fault)` 为每个层组返回只读布尔掩码，允许**部分行成功交付与 `fault` 共存**；已交付行保证字节完整且不再被写入，而**未交付行可能处于半写入状态，其底层槽位内存必须等待 `Attempt.quiesced() == True` 后方可安全回收复用**。

PR `#20608` 修复了 `RemoteMpiCommSessionServer` 的停机竞态：服务端在等待 Worker 退出期间以 `PENDING_POLL_SECONDS = 0.1` 秒周期持续排空控制套接字并立即回复 `SHUTDOWN_ACK`；客户端 `stop_server_main` 循环重连发送停止请求（`STOP_ACK_TIMEOUT_SECONDS = 1.0`, `STOP_MAX_ATTEMPTS = 300`），直至收到显式 `SHUTDOWN_ACK`。

### 2.3 vLLM `ThreadedWatchDog` 全线程栈转储与空闲静默窗口（PR `#55700`）

PR `#55700` 在 `vllm/utils/watch_dog.py` 中实现了守护线程 `ThreadedWatchDog`（默认 `timeout = 300s`, `check_interval = 10s`）：
- **空闲阻塞窗口挂起（`with wd.disable():`）**：在 `shm_broadcast.py` 的 `MessageQueue.acquire_read` / `acquire_write` 等待循环外包裹 `with get_watch_dog().disable():`，后台检查线程在 `active == False` 期间持续令 `last_feed_time = now`，确保空闲等待完全不消耗超时预算，并在退出 `disable()` 时从零重新计时。
- **单次故障单次转储（One-Shot Dump per Stall）**：仅当 `now - last_feed_time > timeout` 且 `last_timeout_log < last_feed_time` 时调用 `faulthandler.dump_traceback(file=f, all_threads=True)`，同一卡死周期绝不重复刷屏，直至 `WorkerProc._execute_worker_rpc` 恢复并再次调用 `wd.feed()`。

---

## 3. 确定性实验与量化基准验证

运行 `python experiments\trtllm_dkg_dsa_mqa_kv_contract_vllm_watchdog\run.py --check` 验证全部契约与指标：

| 评估维度 | 场景配置 | 基线策略 | 新机制表现 | 核心工程收益 |
|---|---|---|---|---|
| **CuTe DSL `next_n=5` 切分 (`#19976`)** | GB300 (`148 SMs`), `B=32, max_ctx=16384` | `atom=1` 重复扫描 KV `5` 次 (`139` 波) | **`num_kernel_atoms=2, atom=3, offsets=[2,0]`** (`56` 波) | **KV 扫描次数减少 `60.00%`** |
| **CuTe DSL `next_n=7` 切分 (`#19976`)** | GB300 (`148 SMs`), `B=32, max_ctx=16384` (`>=24` 波) | `atom=1` 重复扫描 KV `7` 次 (`194` 波) | **`num_kernel_atoms=3, atom=3, offsets=[4,2,0]`** (`84` 波) | **KV 扫描减少 `57.14%`，实测加速 `1.04x–2.60x`** |
| **CuTe DSL 小波次保护 (`#19976`)** | GB300 (`148 SMs`), `next_n=7, B=1, max_ctx=4096` (`1` 波 `<16`) | 盲目启用宽 Tile 切分导致 `0.87x` 退化 | **自动回退 `fallback_atom1_low_waves`** | **消除小 Batch 宽 Tile 负优化** |
| **KV 传输契约与 MPI 停机 (`#20117/#20608`)** | 2 层组部分交付 + `peer_cancelled` / 2.5s Worker 停机 | 未区分逻辑完成与物理静默 / 停机丢包 | **`quiesced()` 门控内存回收 + `0.1s` `SHUTDOWN_ACK`** | **零半写内存踩踏，停机确认延迟 `<= 0.1s`** |
| **vLLM `ThreadedWatchDog` (`#55700`)** | 590s 空闲等待 + 两次独立 `>300s` RPC 卡死 | 空闲误报或重复转储刷屏 | **空闲 `0` 误报，两次卡死精准生成 `2` 份全线程栈** | **100% 消除空闲假阳性与重复刷屏** |

---

## 4. 生产落地指南与适用边界

- **适用场景**：Blackwell (`sm_100f`) 上 DeepSeek-V3.2 / GLM-5 等 DSA 稀疏注意力在任意 `next_n` 投机长度下的 MQA Logits 计算、TRT-LLM 分离式 KV 传输后端插件开发，以及生产环境 vLLM 多卡 Worker 死锁自动取证。
- **不适用 / 边界条件**：`_pick_dsl_split` 的内核级非均匀切分仅在 `waves_atom1 >= 16` 且 `next_n` 无 $\{2, 3\}$ 精确因子时触发；对于 `next_n = 6` 等可被 3 整除的窗口，直接走 `expand_factor = 2, atom = 3` 的调用方精确展开路径。
