# TensorRT-LLM Prefill 分块多模态调度、Blackwell `SwapsMmaAb` 投机树掩码容量与 KVCM2 渐进式主机内存提交，以及 vLLM `libgomp` Fork 暂停 / ROCm HY-V4 iHC 融合与 SGLang Intel XPU 分离式 KV 传输深挖

> **研究阶段**: Phase 46 (`priority: 55`) · **问题编号**: `q-phase46-trtllm-mm-chunk-spec-mask-kvcm2-vllm-ihc-sglang-xpu` · **确定性复现脚本**: [`experiments/trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu/run.py`](../../experiments/trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu/run.py)

---

## 1. 核心架构问题与一手提交总览

在大规模多模态长上下文服务、Blackwell 投机解码（Eagle/Medusa）、分层 KV 缓存管理（KVCM2）以及跨平台（ROCm / Intel XPU）分布式推理中，TensorRT-LLM、vLLM 与 SGLang 近期修复并优化了五项关键运行时机制：

1. **多模态长请求整单编码预算死锁（`TensorRT-LLM#19884`）**：旧版 `DisaggScheduleSimulator` 与 `MultimodalRuntime` 要求请求的所有多模态项一次性装入 `encoder_max_num_tokens` 与 `EncoderCacheManager` 容量，即使开启了 Chunked Prefill 且前缀 KV 已命中，多图/长视频请求仍会因整单 Token 数超限而被永久推迟或拒绝。
2. **Blackwell 投机树注意力掩码在 `SwapsMmaAb` 下越界写（`TensorRT-LLM#20223`）**：Blackwell 生成阶段 FMHA 可动态选择 `KeepsMmaAb` 或 `SwapsMmaAb` 内核，但 Python 侧缓冲仅按固定 `KeepsMmaAb`（`stepQ=128, stepKv=256`）分配 `spec_decoding_bl_tree_mask`。当内核选择 `SwapsMmaAb`（将 `tileSizeQ` 向上对齐至 32 并按每 Token 独立分块）时，写入量可达旧缓冲的 **`4.0x`**，造成静默 GPU 内存越界破坏。
3. **KVCM2 主机内存层启动同步大页清零阻塞与 `mremap` 地址漂移（`TensorRT-LLM#19650`）**：旧版 `HostMem` 启动时同步提交并锁页全部主机缓存配额，或通过 `mremap` 扩容（可能改变基地址导致外部指针失效）。
4. **vLLM `fork` 子进程继承 `libgomp` 线程池死锁（`vllm#62884`）与 ROCm HY-V4 iHC 跨层带宽瓶颈（`vllm#58509`）**：父进程若在 `os.fork()` 前执行过 PyTorch CPU OpenMP 并行区，子进程首次进入 OpenMP 并行区即永久死锁；HY-V4 的 4 通道独立超连接（iHC）在 ROCm 上分 3 个独立算子执行 `post`、`pre` 与 `RMSNorm`，反复读写 `[T, 4, d]` 状态张量。
5. **SGLang Intel XPU 分离式 KV 传输缺页（`sglang#40861`）与 FlashInfer CUTLASS MoE 调优桶失配（`sglang#43363`）**：Intel XPU 设备虚拟地址位于内核高位空间（Bit 63 置 1），直接注册 RDMA 或使用 `int64` 打包内核会导致传输失败或张量截断；同时 PD 分离 Prefill 节点与 Decode 节点混用同一 `tune_max_num_tokens` 公式。

| 仓库与提交 | 日期 | 核心架构变更 | 实测性能 / 正确性收益 |
|---|---|---|---|
| `TensorRT-LLM@b47eb8fe6b` (`#19884`) | 2026-10-21 | `ContextChunk(start, end)` 先解析 KV 前缀复用，再按分块调度多模态项并引入 `TensorLRUCache.releasable_bytes` | 4 图长请求单步编码预算从 `4096`（被拒）降至 `<= 2048`（全部分块通过），并在分块后即时释放已消费嵌入 |
| `TensorRT-LLM@85b1b8323b` (`#20223`) | 2026-10-19 | `getCustomMaskCapacityUpperBound` 联合遍历 `KeepsMmaAb` 与 `SwapsMmaAb` 全部 Tile 形状并透传 `customMaskCapacity` | 揭示 `SwapsMmaAb` 掩码容量达旧版 `KeepsMmaAb` 公式的 **`4.0x`**（`2,097,152` vs `524,288` `uint32`），消除 GPU 越界写 |
| `TensorRT-LLM@0cd9d0b541` (`#19650`) | 2026-10-19 | KVCM2 `HostMem` 采用 `MAP_NORESERVE` 预留完整虚拟地址并由后台线程按 `2 MiB -> 32 MiB` 粒度渐进提交 | `64 GiB` 配额下启动同步提交量从 `64 GiB` 降至 `32 MiB`（缩减 **`2048x`**），彻底移除 `mremap` 基址漂移风险 |
| `vllm@40210d8317` (`#62884`) | 2026-10-21 | `get_mp_context()` 在 `fork` 前扫描 `/proc/self/maps` 调用 `omp_pause_resource_all(_OMP_PAUSE_HARD=2)` | 回收 `libgomp` 空闲线程，根除多卡 Engine Core `fork` 后首次 CPU OpenMP 并行区死锁 |
| `vllm@833349105b` (`#58509`) | 2026-10-21 | ROCm HY-V4 新增 `triton_ihc_post_pre_rms_norm` 融合内核并放开 AITER MXFP8 `MoEActivation.SILU` | 每层 iHC + RMSNorm 从 3 次 Kernel 发射缩减为 1 次，每 Token HBM 流量从 `128 KiB` 降至 `80 KiB`（**`1.60x`** 带宽缩减） |
| `sglang@8e3e2980db` (`#40861`), `f8fd41c36e` (`#43363`), `2c9d919071` (`#45437`) | 2026-10-21 | Intel XPU `ptr & (1 << 63)` 识别 + Host Staging + 禁用 `int64` MLA 打包；MoE Prefill `min(T * dp, 8192)`；`SGLANG_ZMQ_SOURCE_IP` | 打通 Intel XPU NIXL/Mooncake 分离式 KV 传输，避免 MoE Prefill/Decode 调优桶失配与多网卡 ZMQ 源 IP 漂移 |

---

## 2. TensorRT-LLM：按 Prefill 分块调度多模态编码器与 `SwapsMmaAb` 掩码上界

### 2.1 KV 前缀感知的 `ContextChunk` 多模态预算与 `releasable_bytes` (`#19884`)

在 `TensorRT-LLM@b47eb8fe6b` (`#19884`) 中，调度器将多模态编码预算从“整单粒度”收紧为“当前 Prefill 分块粒度”：
1. **先解析 KV 缓存前缀**：在评估多模态预算前先查询 Radix 树得到已复用的前缀长度 `kv_cached_prefix_tokens`（例如 `1536`），位于 `[0, 1536)` 内的 `img0`（`[256, 1280)`）无需重跑视觉编码器；
2. **按活跃窗口 `[chunk_start, chunk_end)` 计费**：对分块 `[1536, 3584)`，`get_mm_items_for_chunk` 仅选中与之相交且未编码的 `img1` 与 `img2`（共 `2048` Token $\le$ `encoder_max_num_tokens=2048`），分块结束后若条目完全落在该分块内则立即释放；
3. **共享哈希引用的精确可释放字节计算**：新增 `TensorLRUCache.releasable_bytes(keys)`，统计传入 `keys` 中每个 `key` 的出现次数 `count`，仅当 `0 < entry.reference_count <= count` 时才计入 `entry.nbytes`，防止多个在途请求共享同一多模态哈希时虚增可用空间。

### 2.2 Blackwell `SwapsMmaAb` vs `KeepsMmaAb` 投机树掩码容量公式 (`#20223`)

Blackwell 生成阶段自定义树掩码（Eagle / Medusa）包含两类 MMA 布局：
- **`KeepsMmaAb`**：查询头与序列维合并分块，`numTilesQ = ceilDiv(seqLenQ * numHeadsQPerKv, stepQ)`，单序列掩码字数（`uint32`）为：
  $$\text{Size}_{\text{Keeps}} = \left\lceil \frac{L_q \cdot H_{q/kv}}{\text{stepQ}} \right\rceil \cdot \left\lceil \frac{L_q + \text{stepKv} - 1}{\text{stepKv}} \right\rceil \cdot \frac{\text{stepQ} \cdot \text{stepKv}}{32}$$
- **`SwapsMmaAb`**：交换 MMA A/B 操作数以处理小 Q Tile（`tileSizeQRaw` $\in \{8, 16, 32\}$），硬件将 `tileSizeQ` 向上对齐至 `32`（`tileSizeQ = ceilDiv(tileSizeQRaw, 32) * 32 = 32`），且每个查询 Token 独立分块 `numTilesQ = seqLenQ * ceilDiv(numHeadsQPerKv, tileSizeQ * numInstsQ)`：
  $$\text{Size}_{\text{Swaps}} = \left(L_q \cdot \left\lceil \frac{H_{q/kv}}{32 \cdot \text{numInstsQ}} \right\rceil\right) \cdot \left\lceil \frac{L_q + \text{stepKv} - 1}{\text{stepKv}} \right\rceil \cdot \frac{\text{numInstsQ} \cdot \text{numInstsKv} \cdot 32 \cdot \text{tileSizeKv}}{32}$$

在典型配置 `maxBatchSize = 64, maxSeqLenQ = 64, numHeadsQPerKv = 8` 下，旧版 Python 代码仅按 `KeepsMmaAb`（`stepQ=128, stepKv=256`）分配 `524,288` 个 `uint32`（`2 MiB`），而 `SwapsMmaAb` 实际需要 **`2,097,152` 个 `uint32`（`8 MiB`，恰为旧版的 `4.0x`）**。`getCustomMaskCapacityUpperBound` 通过遍历全部合法 `(stepQ, stepKv, tileSizeQ)` 取最大值，并在 `runPrepareCustomMask` 入口强制校验 `required <= runnerParams.customMaskCapacity`。

---

## 3. KVCM2 渐进式主机内存层、vLLM `libgomp` Fork 暂停与 SGLang Intel XPU

### 3.1 KVCM2 `MAP_NORESERVE` 固定基址与幂次扩展提交粒度 (`#19650`)

`TensorRT-LLM@0cd9d0b541` (`#19650`) 将 KVCM2 主机内存层重构为“虚拟空间一步到位 + 物理页面后台渐进提交”：
1. **零移动基址预留**：使用 `mmap(nullptr, vmSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)` 一次性预留 `max(maxQuota, alignedSize)` 虚拟地址空间，基地址永不改变，彻底淘汰 `mremap`；
2. **随配额自适应的提交单元**：定义二级层分配粒度
   $$\text{tierAllocationGranularity}(Q) = (2\text{ MiB}) \ll \min\!\left(4,\, \lfloor \log_2(Q / 1\text{ GiB}) \rfloor\right) \in [2\text{ MiB},\, 32\text{ MiB}]$$
   且单次提交步长 $\text{hostCommitUnit} = \text{roundUp}(\max(\text{tierUnit}, \text{slotSize}), \text{slotSize})$。在 `64 GiB` 配额、`16 MiB` Slot 下，启动时仅同步提交首个 `32 MiB` 单元（同步阻塞量缩减 **`2048x`**），其余由后台线程异步 `mprotect(PROT_READ | PROT_WRITE)` + `mbind` + `cuMemHostRegister` 填充。

### 3.2 vLLM `omp_pause_resource_all` (`#62884`)、ROCm HY-V4 iHC 融合 (`#58509`) 与 SGLang Intel XPU (`#40861`)

- **vLLM `libgomp` Fork 安全**：`get_mp_context()` 在 `mp_method == "fork"` 时遍历 `/proc/self/maps` 中所有已加载的 `libgomp` 实例并调用 OpenMP 5.0 `omp_pause_resource_all(2)`（`omp_pause_hard`），在 `fork` 前销毁空闲工作线程，使父子进程在后续并行区各自干净重建线程池。
- **vLLM ROCm HY-V4 `triton_ihc_post_pre_rms_norm`**：将前一层 `HYV4HCPostLayer`（通道写回 $y = x + \text{attn\_post}^\top \text{residual}$）、下一层 `HYV4HCPreLayer`（跨通道 RMSNorm + Sigmoid 门控加权规约）与注意力前 `RMSNorm` 融合进单个 FP32 Triton 内核（`_BLOCK_K=1024, _BLOCK_D=1024`），消除中间 `[T, 4, d]` 与未归一化 `[T, d]` 的全局内存往返，每 Token HBM 读写从 `128 KiB` 降至 `80 KiB`（**`1.60x`** 缩减）。
- **SGLang Intel XPU 分离式 KV 与 CUTLASS MoE 调优**：通过 `bool(ptr & (1 << 63))` 识别 XPU 内核空间设备指针，自动分配对齐的 CPU 锁页 Staging 缓冲并在 NIXL 传输复用前调用 `_await_handles()`；同时在 XPU 上禁用 `int64` 步长拷贝内核 `copy_mla_rows_into_pack` 改用逐 Token RDMA 传输，并将 Prefill 节点的 FlashInfer CUTLASS MoE `tune_max_num_tokens` 限制为 `min(prefill_tokens * attn_dp_size, 8192)`。
