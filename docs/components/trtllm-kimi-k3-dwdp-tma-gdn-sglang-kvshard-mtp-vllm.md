# TensorRT-LLM Kimi-K3 DWDP TMA 14 级环形预取与确定性混合 GDN 解码、SGLang EAGLE MTP 共享 KVShard 分配器与分离式写失败 Abort 毒化、vLLM ROCm Kimi-K3 BF16 KDA 状态与专用解码深度解析

> **Phase 61 · Priority 70 · Track B（Serving / Disaggregated PD / TensorRT-LLM / SGLang / vLLM）**
> **双语版本**: [中文正文](./trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm.md) | [English Version](./trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm.en.md)
> **可复现实验**: `python experiments/trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm/run.py --check`

---

## 1. 架构背景与核心问题

在部署万亿参数混合线性注意力 + 稀疏 MoE 模型（如 Kimi-K3、DeepSeek-V4.1）与 PD 分离（Disaggregated Prefill-Decode）架构时，推理引擎面临五大系统级挑战：

1. **Kimi-K3 跨节点专家权重动态预取（DWDP）带宽受限于在途字节数与协作启动阻塞**：在多节点 NVLink Fabric（MNNVL）上执行逐层专家权重异步预取时，普通每线程 `kUnroll = 4` 的寄存器加载受限于 `MaxThreads = 1024, MaxRegs = 64` 的寄存器预算，难以打满长延迟跨机框链路；同时 TRT-LLM 路由算子若默认使用 `cudaLaunchCooperativeKernel`（申请 `smCount - 8` 个驻留 Block），在持久化 DWDP 预取 Block 占用部分 SM 时会被迫串行等待。
2. **混合 Prefill+Decode Batch 下 GDN 解码递归的确定性漂移**：当新 Prefill 请求加入正在运行的 Gated DeltaNet（GDN）解码批次时，若对全部有效 token 统一执行因果卷积后处理并切分，Prefill 样本会改变解码 token 的算子分块边界，违反 `FORCE_DETERMINISTIC=1` 的逐 token 比特级复现要求。
3. **SGLang EAGLE MTP 草稿池缺少 KV Cache 分片（KVShard）支持**：当目标模型开启 `kv_shard_size = N`（`PageInterleavePoolAllocator`）时，EAGLE 多 Token 预测（MTP）草稿池若退回普通 `TokenToKVPoolAllocator`，将导致草稿模型与目标模型页面映射不一致并引发显存超配。
4. **PD 分离传输写失败后的过早 Abort ACK 页面重用风险**：在 SGLang Mooncake/NIXL/MoRI 分离式 KV 传输中，当 Prefill 端遇到 `_NixlPostError` 或 MoRI `status.Failed()` 时，网卡队列中可能仍有尚未落盘的飞线 RDMA 写包。若此时收到 Decode 端 `ABORT` 请求并立即回复 `ACK`，Decode 端将提前回收并重用这些 KV 页面，导致残留 RDMA 写覆盖新请求数据。
5. **vLLM ROCm MI355X Kimi-K3 FP32 KDA 状态显存开销与 DSv4.1 稀疏索引尾部脏值**：Kimi-K3 每层 96 个 `128×128` KDA 递归状态若以 FP32 存储需 `6.0 MiB/层/槽位`；同时 DeepSeek-V4.1 `combine_topk_swa_indices` 在复用调用方预分配缓冲区时未将 `>= valid_len` 的尾部槽位重置为 `-1`。

---

## 2. 上游源码级实现与公式推导

### 2.1 TensorRT-LLM Kimi-K3 DWDP TMA 14 级环形预取与混合 GDN 确定性拆分（`3024eb446e72`, `34e813733b18`）

1. **寄存器 `kUnroll = 8` 与 TMA `cp.async.bulk` 14 级共享内存环（`dwdpKernels.cu`）**：
   - **寄存器内核**：保持 `__launch_bounds__(1024, 1)` 与 64 寄存器上限不变，将每线程循环展开度从 `kUnroll = 4` 提升至 `kUnroll = 8`（每线程单批发出 $8 \times 16\text{ B} = 128\text{ B}$ `ld.volatile.global.v4.u32`），在相同驻留 Block 数下实测跨节点读取吞吐达到 `kUnroll = 4` 的 **~3.0x**。
   - **TMA Bulk-Copy 环形内核（`dwdpGatherFabricTmaKernel`）**：在 SM90+ 上由单线程驱动 TMA 引擎，完全绕过每线程寄存器文件限制。配置为：
     $$k_{\text{stages}} = 14, \quad B_{\text{stage}} = 16,384\text{ B}\ (16\text{ KiB}), \quad B_{\text{smem}} = 14 \times 16,384 + 128 = 229,504\text{ B}\ (\approx 224.1\text{ KiB})$$
     通过延迟补充（Lazy Refill，在发起第 $i$ 阶段读取时仅等待 `mbarrier` 释放至最多 `kStages - 1 = 13` 个未完成阶段），单 SM 持续保持：
     $$B_{\text{in-flight}} = (14 - 1) \times 16,384 = 212,992\text{ B}\ (208\text{ KiB})$$
     的跨节点 NVLink Fabric 在途字节数。
   - **路由协作启动逃生阀**：新增 `TLLM_ROUTING_DISABLE_COOP_LAUNCH=1`（覆盖 DeepSeek-V3 與エクスポート路由内核），在 `smCount = 148`、协作网格要求 `140` 个驻留 Block 而 DWDP 预取已占用 `16` 个 SM（$140 + 16 = 156 > 148$）时，将 `launchClusterKernel` 切换为普通非协作启动，允许与持久化 DWDP 预取并行重叠。
2. **确定性混合 Batch GDN 解码拆分（`gated_delta_net.py`）**：
   - 当 `num_prefill > 0 and num_decode > 0 and not is_target_verify and FORCE_DETERMINISTIC == "1"` 时，启用 `split_decode = True`：先将 `conv_output` 拆分为 `conv_output_prefill` 与 `conv_output_decode`，分别独立调用 `rearrange`、`split` 与 `fused_recurrent_gated_delta_rule_update`，使混合批次中的解码递归输出与纯解码批次保持 **0.0 漂移**；同时通过 `@tensor_cache` 缓存 `_prefill_query_start_loc` 避免每层重复 `torch.cat`。

### 2.2 SGLang EAGLE MTP 共享 KVShard 分配器、分离式写失败毒化与非扩展 IPC 标志（`79ab8f6d7c8f`, `6e667c431009`, `4f47c48b332f`, `6c844eb12ee2`）

1. **EAGLE MTP 草稿池共享 `PageInterleavePoolAllocator`（`79ab8f6d7c8f`）**：
   - 在 `ModelRunner._build_mtp_kv_shard_pool()` 中，MTP 草稿池直接复用主模型的 `self.token_to_kv_pool_allocator`（要求草稿与目标层均为全注意力且同构页大小），并校验：
     $$\text{allocator.size} == \text{draft\_pool.size} \times \text{allocator.shard\_size}$$
   - 逻辑 token 位置 $\text{loc}$ 到物理分片 rank $r \in [0, N)$ 与本地物理行 $\text{local\_row}$ 的双射公式为：
     $$\text{loc} = Q \cdot (N \cdot p_s) + r \cdot p_s + o \iff r = \left\lfloor \frac{\text{loc} \bmod (N \cdot p_s)}{p_s} \right\rfloor,\ \text{local\_row} = Q \cdot p_s + (\text{loc} \bmod p_s)$$
     在 $N = 4, p_s = 16$、2 个完整条带（128 个逻辑 token）下实现 100% 精确双射。
2. **PD 分离写失败延迟 Abort 毒化（`6e667c431009`）**：
   - 在 `MooncakeKVManager` 中新增 `_deferred_ack_poisoned_rooms: set[int]`。一旦 `_NixlPostError` 抛出或 MoRI `transfer_sync_write` 返回 `status.Failed()`，立即将该请求对应的 `bootstrap_room` 加入 `_deferred_ack_poisoned_rooms`。
   - 当后续或并发的 `ABORT` 请求到达时，若 `room in _deferred_ack_poisoned_rooms`，拒绝发送延迟 `ACK` 并打印告警，迫使 Decode 端等待超时后再回收 KV 页面，彻底杜绝飞线 RDMA 写污染新分配页面。
3. **N-Rank 扩散 IPC 标志非扩展内存分配（`4f47c48b332f`）与 DSv4.1 SM90 Prefill 图重放（`6c844eb12ee2`）**：
   - 在 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 下，导出 CUDA IPC 句柄会走 `cudaMemExportToShareableHandle`（文件描述符类型），跨进程导入需调用 `pidfd_getfd`（系统调用号 `438`），在无 `SYS_PTRACE` 容器中触发 `EPERM` 并在 GB200/GB300 上导致 `empty_cache()` 崩溃。`ipc_shareable_zeros()` 临时切换 `torch.cuda.mem_pool_context(None)` 分配普通 `cudaMalloc` 缓冲区。
   - 在 DeepSeek-V4.1 FlashInfer MLA 后端中，当 SM90 既未开启 `c4` 也未开启 `c128` 稀疏路径（`not (has_c4 or has_c128)`）时，解除 `16384` 上下文截断限制，允许 `32768+` 稠密 Prefill 直接重放 CUDA Graph。

### 2.3 vLLM ROCm Kimi-K3 BF16 KDA 状态、MI355X 专用解码与 DSv4.1 哨兵填充（`dc05b1455636`, `098eb89ea45c`, `9bd5c2a0582d`）

1. **Kimi-K3 BF16 KDA 递归状态与向量化写回（`dc05b1455636`）**：
   - 在 `fused_recurrent.py` 与 `fla/ops/kda/gate.py` 中支持 `state_dtype = torch.bfloat16`，通过 `bf16x4` 向量化存储 `store_state4` 直接读写 `[num_Spec, 96, 128, 128]` 状态张量。单层单槽位显存从 $96 \times 128 \times 128 \times 4 = 6,291,456\text{ B}$ (`6.0 MiB`) 降至 $3,145,728\text{ B}$ (`3.0 MiB`)，**节省 50.0% KDA 状态池显存**。
2. **ROCm MI355X 专用解码（`q=8`）与 MoRIIO 角色护栏（`098eb89ea45c`）**：
   - 在 MI355X（`TP8`, 256 CUs）上为 Kimi-K3 Mla 启用 `num_speculative_tokens = 7`（查询长度 `q = 8`）专用解码内核，并在 `MoRIIOConnector` 中允许 `kv_consumer` 端启用专用解码，同时显式拦截不支持的 `kv_producer` 端配置。
3. **DSv4.1 预分配稀疏索引 `-1` 哨兵初始化（`9bd5c2a0582d`）**：
   - 在 `combine_topk_swa_indices` 收到外部 `out=(combined_indices, combined_lens)` 时，先执行 `combined_indices.fill_(-1)`，消除 `[valid_merged_count, topk + swa_cap)` 尾部槽位的陈旧垃圾索引。

---

## 3. 确定性实验结果与对比矩阵

运行 `python experiments/trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm/run.py --check` 得到的验证结果如下：

| 评估维度 | 修复 / 优化前基线 | Phase 61 修复 / 优化后 | 关键量化收益 / 架构不变式 |
| :--- | :--- | :--- | :--- |
| **TRT-LLM DWDP 跨节点预取在途字节数** | 寄存器 `kUnroll=4` (`64 B/线程`) | **寄存器 `kUnroll=8` (~3.0x 速率) / TMA 14 级环 `212,992 B/SM`** | 单 SM 保持 `13 × 16 KiB = 208 KiB` 异步 TMA 在途流量 |
| **TRT-LLM 混合 Batch GDN 解码确定性** | Prefill+Decode 混合切分导致 `1.8e-3` 漂移 | **`split_decode=True` 独立解码后处理** | **`0.0` 数值漂移** + `@tensor_cache` 零重复拼接 |
| **SGLang EAGLE MTP 草稿 KV 池分配** | 独立非分片分配器，无法对齐 `kv_shard_size=4` | **共享 `PageInterleavePoolAllocator` (`262,144` 逻辑槽位)** | **128/128 双射零冲突**，草稿与目标池严格同构 |
| **SGLang PD 分离写失败 `ABORT` 处理** | 立即发送延迟 `ACK`，Decode 提前重用页面 | **`_deferred_ack_poisoned_rooms` 拦截 `ACK`** | **100% 阻断飞线 RDMA 写覆盖**新请求 KV 页 |
| **vLLM Kimi-K3 KDA 递归状态显存 (`96×128×128`)** | FP32 状态：`6.0 MiB` (`6,291,456 B`) / 层 / 槽位 | **BF16 状态 (`bf16x4`)：`3.0 MiB` (`3,145,728 B`)** | **KDA 状态显存降低 50.0%**，容量翻倍 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **计算练习**：在 Kimi-K3（48 层 KDA，每层 96 个 `128×128` 头）中，若并发缓存 256 个请求的递归状态（含 8 个投机草稿槽位/请求，共 $256 \times 8 = 2048$ 个状态槽位），将 KDA 状态从 FP32 切换为 BF16 可在整个 TP8 节点上节省多少 GiB 显存？
   - *解答*：全模型单槽位字节数从 $48 \times 6\text{ MiB} = 288\text{ MiB}$ 降至 $144\text{ MiB}$。在 2048 个槽位下，总显存从 $2048 \times 288\text{ MiB} = 576\text{ GiB}$ 降至 $288\text{ GiB}$，整机节省 **288 GiB**（每张 TP8 卡节省 **36 GiB**）。
2. **可证伪假设**：在 PD 分离推理中，当 Prefill 端调用异步/同步 RDMA Write 返回错误码后，立即通知 Decode 端释放目标 KV 页面是否安全？
   - *证伪验证*：不安全。RDMA 网卡重传队列或 PCIe 写通道中可能仍有属于该 WQE 的在途数据包；在连接被销毁或超时排空前提前把物理页分给新请求，会导致静默的跨请求 KV 内存污染。
