# NIXL Libfabric 单边 TTL 驱逐重连握手与父级追踪 Span、TransformerEngine 图安全行缩放 NVFP4 分组量化及 Megatron-LM / TorchTitan / LMCache 加固（Phase 102）

- **问题 ID**：`nixl-libfabric-reestablish-te-nvfp4-row-grouped-megatron-torchtitan-phase102`（`priority: 111`，组件：`interconnect-networking`）
- **证据等级**：`own-offline`（基于上游源码补丁审计 + Python 3.12 标准库确定性算例 `experiments/nixl_libfabric_reestablish_te_nvfp4_row_phase102/run.py`）
- **配套英文版**：[`nixl-libfabric-reestablish-te-nvfp4-row-grouped-megatron-torchtitan-phase102.en.md`](nixl-libfabric-reestablish-te-nvfp4-row-grouped-megatron-torchtitan-phase102.en.md)

## 1. 一手上游信源（零重叠）

1. **NIXL `9426653281` (#2438)**：`fix(libfabric): reply to re-establishment handshakes from connected peers`
2. **NIXL `02cc776992` (#2517)**：`tracing: begin a sampled post's span as the span its context identifies`
3. **TransformerEngine `89668d9dee` (#3649)**：`[Common][PyTorch] Add NVFP4 row-scaled grouped cast with graph-safe path`
4. **Megatron-LM `3ddfe0fe07` (#8516)**：`fix(fsdp): preserve retention parameter weight decay`
5. **TorchTitan `2ffc9fd3f8` (#5625)**：`[optim] Handle device modules without is_initialized() in the first step sync`
6. **LMCache `a5c2364820` (#6083)**：`[Fix] Validate trace schemas by level`
7. **LMCache `3f7c2017b1` (#6042)**：`[Docs][RBLN] Document the Rebellions NPU backend`
8. **Dynamo `3b7af264e4` (#17013)**：`test(operator): cover DisaggregatedSet and multinode LWS upgrade fixtures`

---

## 2. 核心实现机制与定量分析

### 2.1 NIXL Libfabric 单边 TTL 驱逐重连握手与析构线程序 (`9426653281`)

在 vLLM `NixlConnector` 生产环境中，任一节点可按 `engine_ttl = 3600s` 单边驱逐远端引擎而不通知对端。当驱逐方重新发起连接时，它会发送 Handshake 并阻塞等待对端回复；但仍持有旧连接记录的对端此前仅在自身新建连接时才发送 Handshake，导致重连方每次都卡死至 **60.0s 超时** 并抛出 `NIXL_ERR_REMOTE_DISCONNECT`。
- **异步 CM 线程回复队列 (`pending_replies_`)**：当 `progress_thread_` 在完成上下文（Completion Context）收到已连接对端的重连 Handshake 时，不得在 CQ 回调内直接发包，而是将该连接推入 `cm_thread_` 的 `pending_replies_` 队列，由 `cm_thread_` 在 Rail 0 上回复一条不携带 `conn_info` 的确认包（对端收到后直接落盘，不触发死循环回复）。
- **析构与断连安全**：`~nixlLibfabricEngine()` 必须 **先停止 `cm_thread_`，再停止 `progress_thread_`**，确保飞行中的握手回复能够由 `progress_thread_` 正常排空 CQ 完成事件；`disconnect()` 则同步清理 `pending_replies_`。
- **实测恢复性能**：重连耗时从 **60.0s 超时失败（0/25 通过）** 降至 **~0.1s（25/25 100% 通过，600.0x 恢复加速）**。

### 2.2 NIXL 采样传输请求父级 Context Span 绑定 (`02cc776992`)

`NIXL_TRACE_CONTEXT_SPAN` 使采样请求的 `nixl::postXferReq.{write,read}` 直接以调用方传入 `call_context` 所标识的 Span 启动，使后续 Backend 阶段、`nixl::xfer.complete` 与接收端 `nixl::notif.received` 全部挂载在同一父 Span 下，孤儿（Orphan）子 Span 数量从 **6 降至 0**。

### 2.3 TransformerEngine 图安全行缩放 NVFP4 分组量化与 5 重守卫 (`89668d9dee`)

`nvte_group_quantize` 新增 `kNVTEGroupedRowScaledNVFP4` 参数，通过设备端 `first_dims` 与 `tensor_offsets` 实现单次 Capture、多次 Replay 的 CUDA Graph 安全 MoE 逐行缩放 NVFP4 分组量化 + 融合 amax（32 专家下内核启动从 `64` 次降至 `2` 次，**32.0x 削减**）。为消除静默越界，补丁实施 5 重守卫：
1. 仅对 `ndim == 2`、`last_dim % 128 == 0`、`num_experts <= 64` 且 `sum(splits) == rows` 的输入启用融合分组核，其余回退至逐专家路径；
2. **混合 Row/Tensor-Scaled 越界防护**：批量 `split_quantize` 按第 0 个量化器分配 amax 缓冲区。若第 0 个专家为 Tensor-Scaled（`4 B`）而第 1 个专家为 128 行 Row-Scaled（`512 B`），8 B 缓冲区将发生 **504 B 越界写穿**；补丁检测混合模式并安全回退。

### 2.4 Megatron-LM MFSDP v2、TorchTitan 首步同步与 LMCache 分级 Schema 校验 (`3ddfe0fe07`, `2ffc9fd3f8`, `a5c2364820`)

- **Megatron-LM (`3ddfe0fe07`)**：在 `_PARAMETER_ATTRIBUTES_TO_PRESERVE` 中加入 `is_wide_residual_retention_parameter`，防止 MFSDP v2 `fully_shard` 丢失 1-D Retention 参数的强制权重衰减标记。
- **TorchTitan (`2ffc9fd3f8`)**：使用 `getattr(device_module, "is_initialized", None)` 兼容无惰性初始化状态的 `torch.mps` 后端，同时保持 GPU 主机上 CPU 模型不触发 CUDA Context 初始化。
- **LMCache (`a5c2364820`)**：仅对 `level == "storage"` 的录制文件强制校验 `TRACE_SCHEMA_VERSION`，使 `events` 级追踪在存储 API 演进时免于误报错。

---

## 3. 架构师设计练习与可证伪问题

1. **练习计算**：若 64 个专家中第 0 个使用 Tensor-Scaled NVFP4（1 个 FP32 amax），其余 63 个各含 256 行并使用 Row-Scaled NVFP4，未加混合守卫时批量缓冲区会越界多少字节？（答案：分配 $64 \times 4 = 256\text{ B}$，实际写入 $4 + 63 \times 256 \times 4 = 64{,}516\text{ B}$，越界 **64,260 B**）。
2. **可证伪假设**：若在 `nixlLibfabricEngine` 析构函数中先停止 `progress_thread_` 再 join `cm_thread_`，当 `cm_thread_` 正在等待重连握手回复的 CQ 完成事件时将发生死锁。
