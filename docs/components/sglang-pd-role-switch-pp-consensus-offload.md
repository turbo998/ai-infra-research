# SGLang PD 分离运行时角色无损热切换 (`role_switch.py`)、`PPConsensusStore` 零 Pickle ZMQ 共识复制与解码端增量 KV 异步卸载 (`DecodeKVCacheOffloadManager`) 机制

> **English Version**: [sglang-pd-role-switch-pp-consensus-offload.en.md](sglang-pd-role-switch-pp-consensus-offload.en.md) | **关联实验**: [`experiments/sglang_pd_role_switch_offload_sim/README.md`](../../experiments/sglang_pd_role_switch_offload_sim/README.md)

### 一句话结论

SGLang 在 PD（Prefill-Decode）分离推理架构中实现了三大关键自治机制：（1）**运行时角色无损热切换（`role_switch.py::handle_pd_role_switch`）** 利用与角色无关的共享 Token KV 池（永不重新分配 KV 显存），在通过 7 项表驱动前置检查（`_reject_reason`）、空闲排空门禁（`is_fully_idle()`）与 Decode CUDA Graph 显存余量预检（`available_graph_gb >= decode_cuda_graph_memory_gb`）后，仅用约 **180 ms** 完成传输队列与上下文的原子拆建（相比 42 秒冷重启提速 **233.33x**）；（2）**`PPConsensusStore`（`pp_consensus_store.py`）** 采用 `msgspec.Struct(array_like=True, frozen=True)` 四元数组 MsgPack 协议替代不安全的 Python `pickle`，在彻底封堵未认证 ZMQ PULL 端口远程代码执行（RCE）风险的同时将跨 PP Rank 复制报文体积缩减 **72.31%**（`65 B -> 18 B`）；（3）**解码端增量 KV 异步卸载器（`DecodeKVCacheOffloadManager`）** 自动接管 Prefill 未对齐的 Prompt 尾部 Token（`origin_len - prefill_offloaded_len`），按 `offload_stride` 步长增量异步卸载生成期 KV 页，并通过 `swa_indices > 0` 过滤 `UnifiedSWAKVPool` 的 0 号 Sink 页。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `python/sglang/srt/disaggregation/role_switch.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `cc820a4e586c32d96032dd38735508812d8d0729` | 实现 `run_event_loop_supervisor`、`handle_pd_role_switch`、7 项表驱动 `_reject_reason` 预检、Decode CUDA Graph 显存余量校验与 `teardown_disaggregation`。 |
| `python/sglang/srt/disaggregation/pp_consensus_store.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `5f346a2ff0bc9390fa66d322b1849c688de917d0` | 实现跨 Pipeline Parallel（PP）Rank 向 PP0 后台异步复制元数据的 `PPConsensusStore`，强制使用 `msgspec.msgpack` 定长数组结构 `_ReplicationOp`。 |
| `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `6ee0838bef13f78fe19f36d630ccd91a8a1824a8` | 实现解码侧增量 KV 缓存卸载生命周期管理、`WeakKeyDictionary` 请求状态追踪、Prompt 未对齐尾部补齐与 `UnifiedSWAKVPool` 滑动窗口页地址转换。 |
| `python/sglang/srt/disaggregation/checksum.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `74d5900c9774d61c5c0b42f1ec3c0b48e13b8564` | 提供 PD 分离跨节点 KV 传输的数据完整性校验工具，辅助诊断 RDMA/Mooncake/NIXL 传输损坏。 |

### 核心实现路径、数据布局与同步契约

1. **零 KV 重分配的运行时 Prefill $\leftrightarrow$ Decode 角色热切换（`role_switch.py:37-195`）**：
   - 在 SGLang 中，底层 Token KV Pool（`req_to_token_pool` 与 `token_to_kv_pool_allocator`）是角色无关的；切换角色无需释放或重新分配数十 GiB 的 KV 显存池，仅需销毁旧角色的队列与传输管理器（`teardown_disaggregation`）并重建新角色结构（`init_disaggregation`）。
   - **可恢复拒绝（`safe_to_restore=True`）vs 致命熔断（`_pd_role_switch_unhealthy=True`）**：
     - 在进入实质 `teardown_disaggregation` 前，任何失败（未开启 `--enable-pd-role-switch`、实例非空闲 `!is_fully_idle()`、开启了 `SGLANG_DISAGG_STAGING_BUFFER`、或切换到 `decode` 时 `available_graph_gb < decode_cuda_graph_memory_gb`）都会立即返回 `safe_to_restore=True`，外部路由器可立即将该实例重新纳入原角色流量池。
     - 一旦开始执行 `teardown_disaggregation`，旧角色已被拆除且不支持原地回滚（No In-Place Rollback）；若此时抛出异常，实例置位 `scheduler._pd_role_switch_unhealthy = True` 并返回 `safe_to_restore=False`，强制要求容器重启。而后续的 `ensure_decode_cuda_graphs` 捕获失败则被 try-except 捕获并优雅降级为 Eager 执行。
2. **`PPConsensusStore` 零 Pickle ZMQ 共识复制（`pp_consensus_store.py:25-145`）**：
   - 在多阶段流水线并行（PP）的 PD 分离集群中，`PP > 0` 的各 Rank 需要将KV 传输完成状态或元数据异步汇总至 `PP0`。
   - 由于 `PP0` 的 ZMQ `PULL` 套接字绑定在可路由主机网卡地址且无传输层认证，若使用 Python `pickle` 反序列化将直接暴露任意代码执行（RCE）漏洞。`PPConsensusStore` 定义了 `@msgspec.Struct(array_like=True, frozen=True)` 的 `_ReplicationOp(kind, rank, key, value)`，并在**调用者线程**完成 `_encoder.encode(op)`（确保非法类型在主线程立即报错而不会杀死后台发送线程），解码端严格拒绝任何非四元数组结构的畸形消息。
3. **Decode 端增量 KV 卸载与未对齐 Prompt 尾部接管（`decode_kvcache_offload_manager.py:85-195`）**：
   - Prefill 实例仅卸载页对齐的前缀部分：`prefill_offloaded_len = (len(origin_input_ids) // page_size) * page_size`。
   - Decode 实例在生成过程中计算 `all_tokens = origin_input_ids + output_ids[:-1]` 与 `incremental_total = len(all_tokens) - prefill_offloaded_len`，从而**自然将 Prefill 遗留的未对齐 Prompt 尾部（`len(origin_input_ids) % page_size` 个 Token）与新生成的 Output Token 拼接为连续页**，每满 `offload_stride`（`page_size` 的整数倍）触发一次 D2H 卸载。
   - 针对 `UnifiedSWAKVPool` 混合滑动窗口池，`translate_loc_from_full_to_swa` 将虚拟页映射为物理 SWA 槽位，并通过 `swa_indices[swa_indices > 0]` 剔除编号为 `0` 的特殊 Sink 页；同时使用 `WeakKeyDictionary` 以 `Req` 对象身份（而非可能被快速复用的 `rid` 字符串）作为键，防止异步 D2H 拷贝在途时发生 `rid` 冲突或内存泄漏。

### 关键公式、单位与工程假设

1. **Prefill 与 Decode 协同分页卸载边界公式**：
   $$L_{\text{prefill\_off}} = \left\lfloor \frac{L_{\text{prompt}}}{P_{\text{page}}} \right\rfloor \cdot P_{\text{page}}, \quad S_{\text{stride}} = \max\left(P_{\text{page}}, \left\lfloor \frac{S_{\text{env}}}{P_{\text{page}}} \right\rfloor \cdot P_{\text{page}}\right)$$
   $$L_{\text{inc\_total}}(t) = (L_{\text{prompt}} + t - 1) - L_{\text{prefill\_off}}, \quad \Delta L_{\text{aligned}}(t) = \left\lfloor \frac{L_{\text{inc\_total}}(t) - L_{\text{inc\_done}}}{S_{\text{stride}}} \right\rfloor \cdot S_{\text{stride}}$$
   其中 $L_{\text{prompt}}$ 为原始 Prompt Token 数，$t$ 为当前 Decode 步数，$P_{\text{page}}$ 为页大小，$S_{\text{stride}}$ 为实际卸载步长。
2. **Decode CUDA Graph 切换前显存准入条件**：
   $$\text{WillCapture} = (\text{new\_role} == \text{"decode"}) \land (\neg \text{HasDecodeGraphs})$$
   $$\text{WillCapture} \implies \left(M_{\text{req\_graph}} \neq \text{None}\right) \land \left(M_{\text{avail\_gpu}} \ge M_{\text{req\_graph}} \ge 0\right)$$

### 确定性离线仿真算例与量化对比

通过运行标准库确定性仿真器 [`experiments/sglang_pd_role_switch_offload_sim/run.py`](../../experiments/sglang_pd_role_switch_offload_sim/run.py)（结果固化于 [`experiments/sglang_pd_role_switch_offload_sim/results.json`](../../experiments/sglang_pd_role_switch_offload_sim/results.json)），得到以下核验数据：

| 评估维度 | 指标 / 场景 | 传统/基线方案 | SGLang 新机制 (`df13d5dce3`) | 量化收益 / 工程结论 |
| :--- | :--- | :--- | :--- | :--- |
| **PD 角色弹性切换耗时** | Prefill $\to$ Decode 角色翻转 | 冷重启进程 + 重载权重 + 重建 KV 池：`42,000 ms` | 原地保留 KV 池热切换：`180 ms` | **233.33x 切换加速**，且 `kv_pool_reallocated = False` |
| **CUDA Graph 显存不足预检** | 需 `4.0 GB`，仅剩 `2.2 GB` | 拆除旧角色后捕获 OOM 崩溃（实例不可用） | 拆建前拦截并返回 `safe_to_restore = True` | 零中断恢复原 Prefill 路由，避免半拆毁僵尸实例 |
| **`PPConsensusStore` 序列化开销** | 跨 PP Rank 复制单条 `_ReplicationOp` | Python `pickle` 字典：`65 Bytes`（存在 RCE 风险） | `msgspec.msgpack` 定长数组：`18 Bytes` | **缩减 72.31% 报文体积**，彻底阻断反序列化 RCE |
| **Decode 增量卸载尾部补齐** | `L_prompt=230, page=64, stride=128` | Prefill 仅卸载 `192` tokens，尾部 `38` tokens 丢失 | Decode 自动从偏移 `192` 起拼接卸载 `[192, 320), [320, 448)` | **100% 接管 38 个未对齐 Prompt 尾部 Token**，无缓存断洞 |

### 对质量、SLO、拓扑、成本与运维的架构影响

1. **潮汐流量下的 GPU 池弹性（SLO 与成本）**：当线上流量从长文本摘要（Prefill 密集）突变为长思维链推理（Decode 密集）时，集群调度器可在 sub-200ms 内将空闲 Prefill 节点原地翻转为 Decode 节点，无需预留双倍冗余机器。
2. **分布式控制面安全（Security）**：在多租户或容器网络中，用 `msgspec.Struct(array_like=True, frozen=True)` 替代 `pickle` 是所有 ZMQ 控制面通信的标准安全范式。
3. **KV 释放时序安全（Race Avoidance）**：`DecodeKVCacheOffloadManager.offload_kv_cache` 明确规定**不在卸载发起时释放 GPU 槽位**（而在请求结束 `_release_finished_req` 时统一释放），因为正在 Decode 的请求后续步仍需通过 `req_to_token` 注意力读取这些槽位，且异步 D2H 拷贝仍在读取显存（`has_inflight_device_transfer`）。

### 架构师解释、计算与设计练习

1. **解释练习**：为什么 `DecodeKVCacheOffloadManager` 使用 `WeakKeyDictionary[Req, OffloadedState]` 而不是 `dict[str, OffloadedState]`（以 `req.rid` 为键）？
   - *参考解答*：在高并发服务中，当一个请求完成或中止后，其字符串 `rid` 可能被后续新请求复用，而前一个请求的异步 D2H 卸载线程可能尚未结束；以 `Req` 实例的引用作为 WeakKey 既能精确隔离不同请求实例的生命周期，又能在 `Req` 被垃圾回收时自动清理字典项而不会钉死内存。
2. **计算练习**：设 `origin_prompt_len = 230`，`page_size = 64`，`offload_stride = 128`。请问在 Decode 的第几步（`step`）会触发第一次和第二次 Decode 侧增量卸载？
   - *参考解答*：`prefill_offloaded_len = (230 // 64) * 64 = 192`，未对齐尾部为 `230 - 192 = 38` tokens。由于 `all_tokens_len = 230 + step - 1`，当 `all_tokens_len - 192 == 128`（即 `all_tokens_len = 320`）时触发第一次卸载 `[192, 320)`，此时 `step = 320 - 229 = 91`；当 `all_tokens_len = 448` 时触发第二次卸载 `[320, 448)`，此时 `step = 448 - 229 = 219`。
3. **设计练习**：全局负载均衡器（Router）在调用 SGLang 实例的 PD 角色切换 API 时，应如何根据返回的 `safe_to_restore` 与 `_pd_role_switch_unhealthy` 设计状态转换？
   - *参考解答*：先将实例从路由表摘流（Drain）等待 `is_fully_idle()`；调用切换后若 `success=True`，将其挂载到新角色池；若 `success=False` 且 `safe_to_restore=True`，立即将其恢复回原角色池继续服务；若 `safe_to_restore=False`，则立即隔离该实例并触发 Kubernetes Pod 重启。

### 可证伪后续问题与核验边界

1. **核验边界**：本分析基于 `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` 一手源码与本仓库确定性离线仿真（`own-offline`），未在多节点 RDMA 集群上实测 Mooncake/NIXL 后端 `km.teardown()` 的真实尾延迟。
2. **可证伪后续问题**：当 `new_role == "decode"` 且尚未预捕获 CUDA Graph 时，若在启动阶段预先捕获好 Decode CUDA Graph 并驻留显存，虽可免去切换时的捕获耗时，但常驻 Graph 私用内存池对 Prefill 角色最大可服务上下文长度的挤压比例是多少？
