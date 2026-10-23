# Megatron-LM mHC/Wide-Residual 统一重计算规划与流式激活淘汰切片、TransformerEngine 寄存器常驻 MXFP8 (`exp2f_rcp_2x` 254/255 编码与 Swizzled Scale)、TorchTitan Sticky Session `repin_load_ratio=1.5` 与 LMCache/Mooncake/NIXL/Dynamo 传输护栏深度解析

> **语言 / Language**: [中文](megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake.md) | [English](megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake.en.md)  
> **Phase**: 70 (`priority: 79`, Track C — Distributed Training, RL Rollout Routing & KV/RDMA Storage Plane)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `author-benchmark` (作者 TE Blackwell MXFP8 与 TorchTitan RL CPU 仿真数据) + `own-offline` (本项目确定性标准库仿真 `experiments/megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake/run.py`)

---

## 1. 架构背景与核心工程问题

在分布式训练（Megatron-LM、TransformerEngine、TorchTitan）与 KV Cache/推理控制面（LMCache、Mooncake、NIXL、Dynamo）的最新演进中，六个跨层一致性与性能瓶颈被集中攻克：

1. **Megatron-LM mHC 与 Wide-Residual 统一激活重计算边界规划 (`NVIDIA/Megatron-LM@c13b6e74e86d`, `#7923`) 与流式激活淘汰切片对齐 (`@a54dd7378c25`, `#8150`)**：在 Multi-Head/Manifold Hyper-Connections (mHC) 与 `FullLayerSpec` 宽残差（Wide-Residual）架构中，若激活重计算切分点恰好落在不可分割的 `atomic_layer_pairs`（如 `(Attention, MLP)` 宽残差对）中间，为何会导致反向重放状态断裂？在基于内存剖析的激活淘汰（Activation Eviction）中，为何合并段（Merged Segments）跨越淘汰检查点边界时会导致 `_try_send_streaming_partials` 流式分片失步？
2. **TransformerEngine 寄存器常驻 MXFP8 Rowwise/Bidimensional Cast-Only 算子与 E8M0 `254/255` 边界编码修复 (`NVIDIA/TransformerEngine@9233c73a1b6e`, `#3459`)**：如何利用 256-bit `ld.global.nc.v8.b32` 消除 TMA 与共享内存数据中转？为什么 `exp2f_rcp_2x` 的指数减法公式 `0x7F007F00 - (biased_exp << 7)` 在 E8M0 码值 `254` ($2^{-127}$ 次正规倒数) 与 `255` ($\text{NaN}$) 上会算出 `0x0000` 和跨通道借位的错误因子？如何在不让载荷触碰共享内存的前提下写出 `128x4` (512B) GEMM-Swizzled 缩放矩阵？
3. **TorchTitan RL Sticky Session `repin_load_ratio=1.5` 慢节点逃逸 (`pytorch/torchtitan@50474ad6037c`, `#5219`) 与 Verifiers 0.4.0 超时奖励隔离 (`@4103c7291ffe`, `#5456`)**：在多轮 Agentic RL Rollout 中，粘性会话路由（Sticky Session Routing）为了复用 Prefix KV Cache 将同一会话永久绑定在首个 Generator 上；当某个 Generator 降速 `2x` 时，如何通过相对均值负载比阈值 `repin_load_ratio=1.5` 回收 **`8.0%–13.7%`** 的端到端墙钟时间？为什么升级 Verifiers 0.4.0 后必须优先检查 `trace.is_timeout or verifiers_episode.is_timeout`？
4. **LMCache NIXL L2 适配器分配槽位容量上报 (`LMCache/LMCache@46e33e85a56d`, `#5800`) 与 InProcess 协议/ZMQ 路径收敛 (`@10dd1e6a124d`, `#5979`)**：当单个 POSIX 文件包含多个页面（`pages_per_file > 1`）时，为什么 `report_status()` 返回 `self._config.pool_size` 会造成 `pool_free_slots > pool_size` 的监控悖论？
5. **Mooncake TENT `io_uring` 短读写完成校验 (`kvcache-ai/Mooncake@4012f2a56f69`, `#4618`) 与 `USE_ASCEND_DIRECT` `:npu_x` 后缀剥离 (`@d04eaf621bce`, `#4627`)**：为什么 `cqe->res >= 0` 不能直接标记 `COMPLETED` 并按 `request.length` 拷贝中转缓冲区？为什么 `USE_ASCEND_DIRECT` 未剥离 `:npu_x` 会触发 `EADDRINUSE` (`503900`) 端口冲突？
6. **NIXL 双进程 UCX 分布式 Trace ID 传递验证 (`ai-dynamo/nixl@13a896bde498`, `#2404`) 与 Dynamo `/health?instances=true` 按需实例序列化 (`ai-dynamo/dynamo@a789babbe37a` `#16454`, `@2ae8a46299b1` `#16633`)**：为什么高频 K8s readiness 探针调用 `/health` 不能默认序列化全量发现实例？当 TensorRT-LLM MPI 子进程崩溃时，Dynamo 监控线程为何必须跳过死锁的 `engine.shutdown()` 直接在守护线程限时退出？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 Megatron-LM `build_recompute_block_end_plan` 与 `_slice_merged_segments` (`#7923`, `#8150`, `#7926`)

1. **统一 mHC 与宽残差重计算规划 (`#7923`)**：
   - 在 `megatron/core/transformer/transformer_block.py` 中，新增纯函数 `build_recompute_block_end_plan(num_layers, recompute_method, recompute_num_layers, atomic_layer_pairs)`，统一驱动 `_checkpointed_forward` 的检查点分块与 mHC 重计算边界信号：
     - 对于 `uniform` 模式：当第 $i$ 层达到目标块长 $(i - \text{block\_start} + 1) \ge K$ 时，若 $(i, i+1) \in \text{atomic\_layer\_pairs}$（例如宽残差不可拆分的 `(Attention, MLP)` 层对），则禁止在 $i$ 处截断，顺延至 $i+1$ 才置 `plan[i+1] = True`；
     - 对于 `block` 模式：前 `recompute_num_layers` 层逐层置 `True`，但跳过任何会拆散 `atomic_layer_pairs` 的边界，并强制 `plan[-1] = True`。
2. **流式激活淘汰段切片对齐 (`#8150`) 与移除 `disable_ep_consensus` (`#7926`)**：
   - 在 `megatron/core/pipeline_parallel/eviction_schedules.py` 中，`_merge_consecutive_layers` 会将连续同类型层合并为单个大段以加速重计算；若某合并段跨越了检查点淘汰边界（eviction boundary），旧版前向直接按整段执行，导致 `_try_send_streaming_partials` 无法在检查点边界及时发出流式部分激活。新增 `_slice_merged_segments` 按淘汰边界精确切分合并段，恢复段级流式发送对齐。
   - `#7926` 彻底移除了冗余的 `disable_ep_consensus` 开关，强制 EP 组内保持一致共识。

### 2.2 TransformerEngine 寄存器常驻 MXFP8 Cast-Only 算子与 E8M0 边界修复 (`#3459`)

在 `transformer_engine/common/cast/mxfp8/specialized/{cast_rowwise,cast_bidim}.cuh` 与 `util/ptx.cuh` 中，TE 为 `sm_100+`（Blackwell 及后续架构）且满足 32B 指针对齐的 BF16 Cast-Only MXFP8 路径引入了寄存器常驻内核：

- **零载荷共享内存与闭式 E8M0 指数推导**：
  - **Rowwise 内核**：每 2 个线程协作处理一个 32 元素 MX 块（每线程执行一次 256-bit `ld.global.nc.v8.b32` 加载 16 个 BF16 = 32B，计算完成后执行一次 128-bit 存储写出 16 个 FP8）。
  - **Bidimensional 内核**：每个 CTA（8 warps = 256 threads）负责 $32 \text{ rows} \times C$ 列，32 行恰好等于列向 MX 块高度，因此每个数据块仅从 HBM 读入寄存器**一次**即可同时完成行向（Warp 内 shuffle 归约）与列向（仅通过数百字节 SMEM 归约列最大值）双向 MXFP8 量化，彻底省去了 TMA 二次搬运流水线（移除冗余软件 L2 预取并拓宽 Tile 后，双向算子相对 TMA 基线的几何平均耗时比从 `0.957` 进一步降至 **`0.864`**；Rowwise 算子合并尾部循环后在 `8160x1920` 上从 `10.66 us` 降至 **`8.96 us` (`-15.9%`)**，在 `4064x4480` 上从 `11.10 us` 降至 **`9.47 us` (`-14.7%`)**）。
  - **BF16 闭式 E8M0 Scale 推导**：对于有限 BF16 幅值 $\text{amax} = m \cdot 2^{e-127}$，满足 $\text{amax} / 2^{E-127} \le 448$ 的最小幂指数在 $m \le 1.75$（7 位尾数 $\le 96$）时为 $e - 8$，在 $m > 1.75$（尾数 $\ge 97$）时为 $e - 7$。由于 $97 + 31 = 128$ 恰好向指数位进 1，直接用整数加法 `(bits + (31 << 7)) >> 7 - 8` 即可无浮点乘法求得精确 E8M0 字节（遍历全部 `32,768` 个 BF16 幅值 100% 比特精确）。
- **`exp2f_rcp_2x` 与 `mx_scale_byte_x2` 的 `254` (`Inf`) / `255` (`NaN`) 编码修复**：
  - 打包倒数函数 `exp2f_rcp_2x(biased_exp)` 通过 `0x7F007F00u - biased_exp * 0x00800080u` 同时在两个 BF16 通道构造 $2^{127 - \text{biased\_exp}}$。
  - **Bug 根因**：该减法仅对 $\text{biased\_exp} \le 253$（正规 BF16 倒数）成立！当块内出现 `Inf` 时，$\text{biased\_exp} = 254$，所需倒数为 $2^{-127}$（BF16 次正规数 `0x0040`），减法却算出 `0x0000`（把整个块乘零抹除）；当 $\text{biased\_exp} = 255$（`NaN`）时，低 16 位通道减法产生下溢向高 16 位借位，产出两通道不一致的垃圾值 `0xFF7FFF80`！
  - **修复**：当 `biased_exp >= 254` 时直接返回常量对 `(biased_exp == 254) ? 0x00400040u : 0x7FFF7FFFu`（PTX 编译为无分支谓词 `MOV/SEL`）；同时在 `cast_bidim.cuh` 中将 `254` (Inf) 与 `255` (NaN) 拆开，为 `mx_scale_byte_x2` 每个半字预留进位位防止借位污染邻列，并用 `(mag + 0x0080)` 的 bit 15 检测替代吞没 NaN 的 `max.xorsign.abs`。
- **4-Warp 协作写出 512B GEMM-Swizzled Scale Tile**：
  - GEMM-swizzled 缩放布局将 $128 \text{ rows} \times 4 \text{ scale cols}$ 打包为连续 512B。`cast_rowwise.cuh` 强制 256 线程（4 个 64-thread quad），将相距 32 行的 4 个子行分摊给 4 个 Warp，各 Warp 保持连续行读取，仅将单字节 E8M0 scale 汇集到几百字节 SMEM 后以 16B 向量写出，载荷全程不进 SMEM。

### 2.3 TorchTitan Sticky Session `repin_load_ratio=1.5` (`#5219`) 与 Verifiers 0.4.0 (`#5456`)

1. **`StickySessionRoutingStrategy` `repin_load_ratio=1.5` (`#5219`)**：
   - 在 `torchtitan/rl/distributed/routing/strategies.py` 中，已绑定候选节点 $c_{\text{pin}}$ 的会话在下一轮路由前计算全体候选的平均在途负载 $\bar{L} = \frac{1}{N}\sum_{h} L(h)$：
     $$
     \text{overloaded} = \left(L(c_{\text{pin}}) > \text{repin\_load\_ratio} \times \bar{L}\right), \quad \text{repin\_load\_ratio} = 1.5
     $$
   - 若 `overloaded` 为真，则回退至 `LeastLoadedRoutingStrategy` 选取最闲节点并将该会话**永久重绑定（re-pin）**至新节点（例如负载 `[6, 2, 2, 2]` 均值 `3.0`，$6 > 4.5 \implies$ 迁移；`[4, 3, 3, 2]` 均值 `3.0`，$4 \le 4.5 \implies$ 保持粘性复用 KV Cache）。在存在 1 个 `2x` 慢节点的 CPU 仿真中，墙钟时间缩短 **`8.0%`**（KV 受限场景缩短 **`13.7%–15.2%`**）。
2. **Verifiers 0.4.0 超时隔离 (`#5456`)**：
   - Verifiers 0.4.0 在阶段超时后仍可能保留 `ok=True` 并设置 `trace.is_timeout=True`。`VerifiersRollouter.rollout_status` 优先判定 `if trace.is_timeout or verifiers_episode.is_timeout: return RolloutStatus.ERROR_TIMEOUT`，并通过 `if status.is_error():` 阻断向超时轨迹发放训练奖励。

### 2.4 LMCache NIXL 槽位容量 (`#5800`/`#5979`)、Mooncake `io_uring` 短完成校验 (`#4618`/`#4627`) 与 NIXL/Dynamo (`#2404`/`#16454`/`#16633`)

- **LMCache NIXL L2 槽位计数 (`#5800`) 与 InProcess 收敛 (`#5979`)**：`NixlStoreL2Adapter.report_status()` 将 `"pool_size": self._config.pool_size`（文件数）修正为 `"pool_size": pool.total_objs`（总页槽位数 = 文件数 $\times$ `pages_per_file`），修复了多页文件下 `pool_free_slots > pool_size` 的 Bug；`#5979` 将 `RemoteMetadata` 与 `IPC_SOCKET_PATH_MAX_LEN = 107`（防 Linux `sockaddr_un.sun_path` 108B 溢出）统一收敛至 `lmcache.inprocess.v1`。
- **Mooncake TENT `io_uring` 短读写拒绝 (`#4618`) 与 `USE_ASCEND_DIRECT` (`#4627`)**：`IOUringTransport::getTransferStatus` 在 `cqe->res >= 0` 时令 `completed_bytes = static_cast<size_t>(cqe->res)`，仅从 staging buffer 拷贝 `completed_bytes`（保护未读尾部不被脏数据覆盖），且仅当 `completed_bytes == cqe_task->request.length` 时置 `COMPLETED`，否则置 `FAILED`；`#4627` 在 `USE_ASCEND_DIRECT` 下同步剥离 `local_server_name` 的 `:npu_x` 后缀，消除 ADXL 与 `SocketHandShakePlugin` 的 `EADDRINUSE` 端口冲突。
- **NIXL 跨进程 UCX Trace 验证 (`#2404`) 与 Dynamo `/health` / TRT-LLM 守护退出 (`#16454`, `#16633`)**：NIXL 通过双进程 `ucx_trace_receiver` + `libtrace_backend_recorder.so` 验证 `nixl::postXferReq.write` 与接收端 `nixl::notif.received` 共享同一 correlation ID；Dynamo `/health` 默认仅返回 `{"status": "healthy", "stage": stage}`，仅在显式携带 `?instances=true` 时才在异步运行时序列化全量发现实例；Dynamo TRT-LLM 监控在检测到 `engine.get_health_check_fatal_error() is not None` 时直接跳过会卡死在死进程 IPC 上的 `engine.shutdown()`，先关闭 runtime 停止接单再调用 `os._exit(1)`。

---

## 3. 本项目确定性离线实验与验证 (`own-offline`)

实验脚本：`experiments/megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake/run.py`（结果固化于 `summary.json`）：

| 验证子项 | 关键输入 / 参数 | 实测 / 仿真输出 | 架构结论 |
|---|---|---|---|
| Megatron `build_recompute_block_end_plan` | 6 层, `uniform K=2`, `atomic_layer_pairs={(1,2)}` | 无原子对 `[F,T,F,T,F,T]` $\to$ 含 `(1,2)` 顺延为 `[F,F,T,F,T,T]` | 防止重计算边界切开宽残差/mHC 原子层对 |
| TE MXFP8 `exp2f_rcp_2x` 边界编码 | `biased_exp = 254` (Inf), `255` (NaN) | 旧版 `0x00000000` / `0xFF7FFF80` $\to$ 修复后 `0x00400040` / `0x7FFF7FFF` | 修复 `Inf` 块被乘零及 `NaN` 跨通道借位污染 |
| TorchTitan `repin_load_ratio=1.5` & Verifiers | 负载 `[6,2,2,2]` vs `[4,3,3,2]`; `is_timeout=True, ok=True` | `[6,2,2,2]` 重绑定至 `idx=1`; `[4,3,3,2]` 保持 `idx=0`; 状态 `ERROR_TIMEOUT` | 兼顾常规前缀缓存命中率与慢节点逃逸，拦截超时脏奖励 |
| LMCache NIXL & Mooncake `io_uring` | `16` 文件 $\times 4$ 页/文件; `req=4096B, cqe=1024B` | `pool_size=64 >= free_slots=62`; 短读返回 `FAILED` 且保留尾部 `3072B` | 消除监控容量倒挂与短 I/O 静默数据截断 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **数学推导练习**：在 TransformerEngine `cast_bidim.cuh` 中，为什么对于 BF16 正数 $\text{amax}$，“尾数 $m > 1.75$”等价于“7 位存储尾数域 $\ge 97$”，且加上 `31 << 7` 即可精确完成向上取整进位？
   - *解答*：BF16 的隐含整数位为 $1$，7 位尾数域 $f \in [0, 127]$ 表示 $m = 1 + f / 128$。因为 $1.75 = 1 + 96/128$，所以 $m > 1.75 \iff f \ge 97$。在位级别加上 $31$ 后，$f + 31 \ge 128 = 2^7$ 当且仅当 $f \ge 97$，该进位恰好落入 bit 7（BF16 偏置指数的最低位），从而在一条整数加法指令内实现 $\lceil \log_2(\text{amax} / 448) \rceil + 127$。
2. **可证伪假设**：在 Kubernetes 部署 Dynamo Frontend（挂载数百个 Prefill/Decode Worker 实例）时，将 `readinessProbe` 指向带完整拓扑列表的 `/health?instances=true` 是否没有性能代价？
   - *证伪条件*：每次 `/health?instances=true` 都会在 Tokio 异步运行时上遍历并 JSON 序列化所有已发现实例，在大规模集群上单次探针耗时达数十毫秒并挤占请求转发 CPU；默认 `/health` 仅返回常数级 `{"status":"healthy","stage":"ready"}`，必须作为高频探针默认路径。
