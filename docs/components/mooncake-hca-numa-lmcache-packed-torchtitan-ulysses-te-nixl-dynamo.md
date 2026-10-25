# Mooncake HCA-NUMA 亲和与 MemFabric 批量注册、LMCache 紧凑字节流 Token 账本、TorchTitan 816B Ulysses RL 拓扑、TransformerEngine 与 NIXL/Dynamo 深度解析

> **English Version**: [mooncake-hca-numa-lmcache-packed-torchtitan-ulysses-te-nixl-dynamo.en.md](mooncake-hca-numa-lmcache-packed-torchtitan-ulysses-te-nixl-dynamo.en.md)  
> **Phase**: 72 (`priority: 81` · Track C) · **核验日期**: `2026-10-25` (UTC+08:00)  
> **实验脚本与密封输出**: [`experiments/mooncake-hca-numa-lmcache-packed-torchtitan-ulysses-te-nixl-dynamo/run.py`](../../experiments/mooncake-hca-numa-lmcache-packed-torchtitan-ulysses-te-nixl-dynamo/run.py) · [`summary.json`](../../experiments/mooncake-hca-numa-lmcache-packed-torchtitan-ulysses-te-nixl-dynamo/summary.json)

---

## 1. 背景与核心架构问题

在跨节点 KV Cache 存储与传输（Mooncake、LMCache、NIXL）、分布式 MoE/RL 训练（TorchTitan、TransformerEngine）以及分离式推理编排（NVIDIA Dynamo）中，`2026-10-23` 至 `2026-10-25` 的上游演进集中解决了以下五个系统瓶颈：

1. **跨 NUMA 域 RDMA 写入/读取的 PCIe/UPI 穿透与昇腾 NPU 页对齐冲突**：在多网卡（HCA）双路/四路服务器上，若 Mooncake Store 随机分配跨 NUMA 的 DRAM 缓冲段，RDMA Write 会跨越 CPU UPI/Infinity Fabric 总线；同时在昇腾 MemFabric 传输层中，逐个注册紧邻的虚拟内存缓冲区在对齐到 NPU 大页（如 2 MB）后会产生重叠区间，触发驱动注册失败。
2. **LMCache 多进程连接器在长上下文解码下的 $O(N^2)$ Token 列表拷贝膨胀**：在 vLLM Lazy Offload 模式下，每个解码步都会生成一个 `LoadStoreOp`。此前 `LMCacheMPRequestTracker` 每步都重新切片并复制完整的 `list[int]` token 序列，在 `16K+` 上下文下将长序列复制开销直接压在调度器关键路径上。
3. **TorchTitan 816B-A42B 超大 MoE 在 GB300 RL 训练中的专家 FSDP All-Gather 通信墙**：此前 Muse Spark 816B-A42B DAPO RL 配方使用 `256` 张 GB300 (`FSDP(shard)=64, TP=4, EP=4`)，每层前反向都要在 `64` 个 rank 的专家 FSDP 组内执行庞大的权重 All-Gather / Reduce-Scatter。
4. **TransformerEngine `GroupedTensor` Autograd 视图双重挂钩与 Triton 路由容量溢出**：`GroupedTensor` 在 `__torch_dispatch__` 处理 `alias`/`view` 时若直接传入 `requires_grad=True`，会干扰 PyTorch Autograd 在返回后挂载可微视图元数据；且其 Triton `make_row_id_map` 路径在 `num_out_tokens < total_routed` 时未将超容量行标记为 `-1`，存在越界写入风险。
5. **NIXL UCX 端到端通知追踪断链与 Dynamo Worker 静态堆 GC 停顿**：NIXL UCX 通知原本只携带 `tctx` (`traceparent`)，接收端 `nixl::notif.received` 缺少发送端 `xfer_request_id` 与 `xfer_execution`，无法与发送端阶段精确 Join；而 Dynamo 替代 vLLM API Server 的 worker 进程持有约 `100 万` 个静态 Python 对象（tokenizer、config、engine client），每次 Gen-2 GC 扫描都会造成数百毫秒的请求抖动。

---

## 2. 一手源码审计与实现机制

### 2.1 Mooncake `868a6d05abf9` (`#4631`), `dd1217b6c461` (`#4793`) & `6a835cbf934a` (`#4870`)

1. **HCA-Local NUMA 三层感知分配与副本选择 (`#4631`)**：
   - 新增环境变量 `MC_STORE_HCA_AFFINITY=1` 与 `MC_STORE_HCA_NUMA_SEGMENTS=1`（配合 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`）。
   - 存储客户端通过 `Client::DiscoverSegmentLocality` 发现 RDMA 缓冲区所在 NUMA 节点对应的本地 HCA 列表，并以原子发布语义 `BufferAllocatorBase::SetLocality`（单调 `has_locality_` 标志 + `std::atomic_store_explicit`）上报给 Master。
   - **三层分配策略 (`RandomAllocationStrategy`)**：当请求携带 `preferred_hca_names` 时，分配器在选定段内按 `tier = 0, 1, 2` 三轮扫描：
     - **Tier 0**：优先分配与目标 HCA 匹配的本地 NUMA 范围（消除跨 UPI 流量）；
     - **Tier 1**：回退至未上报局部性（`unknown locality`）的传统段；
     - **Tier 2**：最后才使用已知跨 NUMA 不匹配（`known mismatch`）的段。
   - 同时 `BatchPut` / `BatchReplicaClear` 重构为 `ReplicateConfig::ForKeySubset(original_indices)`，确保 `session_ids`、`group_ids` 与 `per_key_preferred_hca_names` 在分片重试时保持严格索引对齐。
2. **RDMA `submitPostSend` 链式 WR 单次时钟读取 (`#4793`)**：
   - 在 `RdmaEndPoint::submitPostSend` 中，同一 `ibv_post_send` 链表发出的所有 Work Request (`wr_count`) 在持有 `lock_` 期间只需调用**一次** `const int64_t post_ts = getCurrentTimeInNano()` 并赋给所有 `slice->ts`，将 32-WR 链的 `clock_gettime` 系统调用从 `32` 次降至 `1` 次。
3. **昇腾 MemFabric 批量内存注册防页重叠 (`#4870`)**：
   - 在 `MemFabricTransport::addMemoryBuffer(std::vector<BufferDesc>&)` 中，不再逐个循环调用单 buffer 注册，而是收集整批 `(addrs, caps)` 一次性调用 `MemFabricUnderApi::SmemTransBatchRegisterMem`，由底层统一对齐并合并位于同一 NPU 物理页内的相邻缓冲区（如 `0x201000` 与 `0x202000`）。

---

### 2.2 LMCache `6913259264db` (`#5150`) & `00f6d71ed1af` (`#4410`)

1. **紧凑 `bytearray` Token 账本消除 $O(N^2)$ 拷贝 (`#5150`)**：
   - 在 `LMCacheMPRequestTracker` 中引入原地增长的 `packed_tokens: bytearray`（每个 token 按 `TOKEN_STRIDE = 4` 字节紧凑编码），并在 `LazyOffloadManager` 中将每个请求的共享账本 `_token_ledgers: dict[str, bytearray]` 绑定到所有挂起的 `LoadStoreOp.token_bytes`。
   - **复杂度对比**：以 `16,384` prompt tokens 加上 `256` 个缓冲解码步为例，旧实现每步复制完整 `list[int]`，累计拷贝 $\sum_{s=1}^{256}(16384 + s) = \mathbf{4,227,200}$ 个整数元素；新实现仅在初始化时打包一次 prompt 并每步追加 1 个 token，累计仅处理 $16384 + 256 = \mathbf{16,640}$ 个元素（拷贝量降低 **254.04x**），共享账本仅占 `66,560 B`。
2. **混合架构 (GDN + Sliding Window + Full Attention) 逐内核组窗口块计算 (`#4410`)**：
   - 在 `KVLayerGroupsManager` 中新增 `get_kernel_group_window_chunks(kernel_group_idx)`：
     $$\text{chunks} = \begin{cases} -1 & \text{if } \text{sw\_size\_tokens} == -1 \text{ or } (\text{full\_sw\_kv} \land \neg\text{recurrent\_state}) \\ \max\left(1, \left\lceil \frac{\text{sw\_size\_tokens}}{\text{chunk\_size}} \right\rceil\right) & \text{otherwise} \end{cases}$$
   - 在 Intel XPU 上，vLLM 将 GDN 混合模型（如 Qwen3.5）的块大小向上对齐至 64 的倍数（`544 -> 576`），LMCache 将 `chunk_size` 同步设为 `576`；即使开启 `full_sw_kv` 将滑动窗口注意力扩为全注意力（返回 `-1`），GDN 循环状态组（`recurrent_state=True`）仍严格保留 `1` 个尾块窗口。

---

### 2.3 TorchTitan `43d1bb98014e` (`#5546`), `89e4339c963c` (`#5565`) & TransformerEngine `2199b7166417` (`#3701`), `c9c92255465b` (`#3347`)

1. **TorchTitan Muse Spark 816B-A42B GB300 RL 训练拓扑重构 (`#5546`)**：
   - 将 Trainer 规模从 `256 GPUs (FSDP=64, TP=4, EP=4)` 重构为 **`128 GPUs (FSDP=8, TP=4, CP=4, EP=16)`**：
     - 引入 `UlyssesVarlenInnerAttention` (`CP=4`) 将 `65,536` 上下文切分为每 rank `16,384` tokens（包含滑动窗口层）；
     - 将专家并行度从 `EP=4` 提升至 `EP=16`，使专家权重的 FSDP 分片组大小从 `64` 骤降至 **`8`（缩小 8x）**，彻底消除主导训练步骤的跨节点大范围专家 All-Gather，同时将训练 GPU 占用减半（Generator 保持 `16 GPUs: DP=4, TP=4, EP=16` 并启用完整 CUDA Graph）。
   - `#5565` 在 `get_cuda_flash_attention_impl()` 中通过 `importlib.util.find_spec` 强制校验 SM90 `FA3` (`flash_attn_interface`) 与 SM100 `FA4` (`flash_attn.cute`) 是否已安装，缺失时立即抛出明确 `ValueError`。
2. **TransformerEngine `GroupedTensor` 与 Triton 容量截断 (`#3701`, `#3347`)**：
   - `#3701` 在 `GroupedTensor.__torch_dispatch__` 创建视图包装子类时固定传入 `requires_grad=False`，交由 PyTorch Autograd 引擎在外部附加 `DifferentiableViewMeta`，并拦截 `aten.record_stream` 转发给所有底层数据与缩放缓冲区（`rowwise_data`、`columnwise_data`、`scale_inv` 等），防止多流异步释放踩内存。
   - `#3347` 在 Triton `_row_id_map_pass_2_kernel` 中增加 `row_id = tl.where(row_id >= num_out_tokens, -1, row_id)`，使 mask-map 路径在专家容量受限时与 CUDA index-map 路径完全一致地丢弃超容量 token（JAX 路径传入 `2**31 - 1` 保持无截断）。

---

### 2.4 NIXL `f3673f6be3f4` (`#2487`) & Dynamo `45cc9e8a1a9e` (`#16106`), `3032e8a703e5` (`#16482`)

1. **NIXL UCX 16 字节 `treq` 追踪关联标签 (`#2487`)**：
   - 在 UCX 通知序列化 (`nixl::ucx::serializeNotif`) 中，除 26 字节 `tctx` 外，新增 16 字节小端序 `treq` 标签（`struct requestTag { uint64_t requestId; uint64_t execution; }`）。接收端 `readRequestTag` 解析后写入 `nixl::notif.received` 事件，使发送端 (`nixl::submit` / `wire.submitted` / `notif.sent`) 与接收端通知记录可通过 `(xfer_request_id, xfer_execution)` 精确关联。
2. **Dynamo vLLM Worker `--freeze-gc-after-init` (`#16106`) 与工具命名规范 (`#16482`)**：
   - 在 `register_vllm_model` 暴露服务路由前调用 `maybe_freeze_gc_after_init`（执行 `vllm.utils.gc_utils.freeze_gc_heap()`），将初始化阶段生成的约 `100 万` 个静态对象移入 Python GC 永久代（仅允许在 `AGGREGATED`、`DECODE`、`PREFILL` 角色启用，拒绝 `ENCODE` 与 `--headless`），消除负载下长达 `200+ ms` 的 Gen-2 GC 停顿。
   - `#16482` 在 `chat_completions.rs` 中放行包含点号 `.` 的工具函数名（如 `"my-tool.v2"`），对齐 MCP / 命名空间工具调用规范。

---

## 3. 合成验证实验与架构师练习

运行本组件配套的验证脚本：

```bash
python experiments/mooncake-hca-numa-lmcache-packed-torchtitan-ulysses-te-nixl-dynamo/run.py --check
```

### 架构师设计练习与可证伪问题

1. **计算题**：在 `65,536` 上下文的流式推理中，若使用 LMCache Lazy Offload 每 `64` 个生成 token 缓冲一次操作，共生成 `2,048` 个 token（32 次操作），求旧版 `list[int]` 方案与新版 `bytearray` 账本方案累计复制的 4 字节整数个数之比。
   - *解答*：旧版累计复制 $\sum_{k=1}^{32}(65536 + 64k) = 32 \times 65536 + 64 \times \frac{32 \times 33}{2} = 2,097,152 + 33,792 = 2,130,944$ 个元素；新版仅打包 $65536 + 2048 = 67,584$ 个元素，比值为 $2,130,944 / 67,584 \approx \mathbf{31.53\times}$。
2. **可证伪假设**：在 Mooncake Store 中开启 `MC_STORE_HCA_AFFINITY=1` 但不开启 `MC_STORE_HCA_NUMA_SEGMENTS=1`，单机双 NUMA 共享单一跨 NUMA 大内存段时能否获得 Tier-0 本地 NUMA 分配保证？
   - *证伪路径*：不能。上游文档 (`#4631`) 明确指出，未绑定单一 NUMA 节点的段无法上报唯一下游 NUMA/HCA 局部性；必须同时开启 `MC_STORE_HCA_NUMA_SEGMENTS=1` 将存储池按 NIC 本地 NUMA 节点拆分为独立范围，才能激活 Tier-0 精确亲和分配。
