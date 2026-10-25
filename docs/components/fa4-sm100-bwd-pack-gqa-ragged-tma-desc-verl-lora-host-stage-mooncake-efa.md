# FlashAttention-4 SM100 反向 Pack-GQA 与不规则 TMA 描述符边界、veRL 单块宿主 LoRA 暂存及 Mooncake EFA 过期对端驱逐审计

> **English Version**: [fa4-sm100-bwd-pack-gqa-ragged-tma-desc-verl-lora-host-stage-mooncake-efa.en.md](fa4-sm100-bwd-pack-gqa-ragged-tma-desc-verl-lora-host-stage-mooncake-efa.en.md)

## 1. 研究背景与核心问题

在 Blackwell SM100 反向注意力训练、RLHF/GRPO 强化学习 MoE LoRA 权重同步以及百卡分离式推理滚动升级（Rolling Upgrade）中，以下四个底层系统问题直接决定了训练吞吐、内存安全与传输可用性：
1. **FA4 SM100 反向 GQA 的 FP32 原子归约开销与长尾串行化**：此前 SM100 反向传播对每个 Query Head 单独运行 tile，并通过 `cp.reduce.async.bulk`（FP32）跨组归约 `dK / dV` 及执行后处理。若将同一个 KV Head 的 `qhead_per_kvhead` 个 Query Head 折叠进 Q tile 的行维度（Pack-GQA），虽然能消除 `dK / dV` 原子归约并让 `dK / dV` 直接驻留 TMEM 写出，但在不规则 varlen MQA（如 1 条 8k 长序列 + 16 条短序列）下，单个打包 CTA 的迭代次数放大 `qh` 倍，导致严重的关键路径串行化（B200 上从 `2297 µs` 退化至 `3946 µs`）。
2. **FA4 不规则 K/V TMA 5-D Wrap-Around 描述符 128 KB 越界预取与网格填充溢出**：此前用于限制 `seqlen_k` 边界的 5-D wrap-around TMA 描述符具有无界内存跨度，硬件 TMA 单元会在部分越界 box 后预取高达 **~128 KB**，当 V/K 位于分配末尾且紧邻未映射页时触发 `cudaErrorIllegalAddress`；同时 dense Q + `seqused_q` 的描述符工作区未计入 varlen 调度器的每 batch 填充（H100 `B=49, seqlen_q=128, H=6` 实际启动 `582` CTAs，远超未填充公式分配的 `302` 槽位）。
3. **veRL MoE LoRA IPC 同步的万级小张量克隆与 Pin 风暴**：在 `update_weights_from_ipc` 同步每专家 MoE LoRA（如 16,000 个 `(16, 1024)` 张量，共 500 MiB）时，旧路径对每个张量执行两次设备端 `clone()`，再由 vLLM `LoRAModel.from_lora_tensors` 逐个 `.to("cpu", dtype).pin_memory()`，耗时长达 `705 ms`。
4. **Mooncake EFA 滚动升级 `fi_av_insert` 静默拒绝与 NIXL SPSC 伪共享**：在 120 Prefill + 120 Decode 节点的滚动重启中，包含端口/时间戳的 `peer_nic_path` 每次重启都在 `peer_map_` 中新建条目而不释放旧条目，耗尽 libfabric 内部 Address Handle (AH) 缓存导致 `fi_av_insert` 返回 0；同时 NIXL `sharedRingBuffer` 的生产者/消费者原子变量位于同一缓存行，每次读写引发跨核缓存行失效。

---

## 2. 第一手源码审计与关键实现路径

### 2.1 FA4 SM100 Backward Pack-GQA 与关键路径启发式 (`Dao-AILab/flash-attention` @ `bc52a40aef`)
审计 `flash_attn/cute/flash_bwd_sm100.py`、`flash_bwd_preprocess.py` 与 `flash_bwd_postprocess.py`：
- **TMEM 直接写出消除 FP32 `dK/dV` 归约**：在 Pack-GQA 模式下，打包序列的第 `r` 行对应位置 `r // qh` 的第 `r % qh` 个 Query Head；每个 KV Head 仅加载一次 K/V，`dK / dV` 在组内全程驻留 TMEM 并走直接存储路径，无需 FP32 `dK/dV` 累加缓冲区、后处理或信号量。
- **双门控启发式 `_bwd_pack_gqa_heuristic`**：
  1. **关键路径门控（Critical-Path Check）**：检查最长打包 CTA 的迭代数 `qh * ceil(seqlen_q / tile_m)` 是否超过单个 cluster 槽位的平均工作量（varlen 下按平均序列长度计算），从而拦截 `MQA 16/1 hd256` `1×8k + 16×short` 的退化（保持 unpacked `2297 µs`，规避 packed `3946 µs` 的 **1.72x** 慢速陷阱）；
  2. **Wave 量化损失门控**：比较 `gain ~ 0.25 * 2048 / seqlen_q` 与网格缩小 `qh` 倍带来的 wave 损失 `(ceil(w) - w) / ceil(w)`。在 B200 (1400 MHz) 上，`hd128` GQA 2k 提速 **+13%..+26%**，`hd256` 2k/8k 提速 **+4%..+11%**。

### 2.2 FA4 每 CTA 设备端 TMA 描述符更新与工作区上界 (`10b74d98fb`)
审计 `flash_attn/cute/flash_bwd_sm100.py` 与 `flash_fwd_sm100.py`：
- **移除 5-D Wrap-Around 描述符**：改用 `cutlass.utils.TensorMapManager`（GMEM 模式），为每个 CTA 分配 128B 描述符槽位（前向 V 1 个，反向 K/V/Kt 3 个即 `384 B/CTA`）。每当 work tile 切换 batch 时，load warp 将描述符的 rows 范围设为序列末尾（`offset_k + seqlen_k` 或 `seqused_k[b]`），并在 dense 与 block-sparse 路径前执行 fence + prefetch，彻底根除 ~128 KB 越界预取。
- **精确填充网格上界**：将描述符工作区大小修正为每 Head `rows // tile + 2 * batch + 1`（在 `B=49, seqlen_q=128, H=6, tile=128` 下分配 `(49 + 98 + 1) × 6 = 888` 槽位，完全覆盖 H100 实际启动的 `582` CTAs），并在 kernel 内增加越界 trap。

### 2.3 veRL 单块宿主 LoRA 暂存 (`3c6d9b43b6`)、Mooncake EFA 驱逐 (`487da6af82`) 与 NIXL SPSC (`a9f1f7a76f`)
- **veRL 单块 Pin 内存视图 (`verl/workers/rollout/vllm_rollout/utils.py` @ `3c6d9b43b6`)**：接收端按 bucket 一次性分配单块 `lora_dtype` 的 pinned CPU 内存，将 IPC 张量直接拷贝/转型进对应切片并交出视图；vLLM 的 `.to("cpu", dtype).pin_memory()` 识别到已是同类型 pinned CPU 张量后零拷贝返回原引用。在 GB300 上，`4,000` 张量（250 MiB）耗时从 `165 ms` 降至 **`35 ms`（4.71x）**，`16,000` 张量（500 MiB）从 `705 ms` 降至 **`131 ms`（5.38x）**。
- **Mooncake EFA Stale Peer Eviction (`487da6af82`) 与 NIXL SPSC (`a9f1f7a76f`)**：Mooncake 在 EFA 重连时主动驱逐 `peer_map_` 中的旧进程端点并释放 libfabric AH 句柄，将 240 节点集群的活跃 AH 稳定限制在 `240`（消除 17 轮滚动升级后 AH 耗尽）；NIXL `cyclic_buffer.h` 对 SPSC 原子索引进行缓存行隔离并缓存对端单调游标，消除每次 push/pop 的跨核缓存失效。

---

## 3. 可复现实验验证 (`fa4_sm100_bwd_pack_gqa_ragged_tma_desc_verl_lora_host_stage_mooncake_efa`)

| 子系统 / 指标 | 修复/优化前 | 修复/优化后 | 验证结果 | 证据 ID |
|---|---|---|---|---|
| FA4 SM100 Bwd Pack-GQA (`hd128` 2k / Ragged MQA) | Unpacked (`410 µs`) / 盲目 Pack (`3946 µs`) | 关键路径门控 + TMEM `dK/dV` (`325.4 / 2297 µs`) | **+26% 提速 & 规避 1.72x 退化** | `src-fa4-sm100-bwd-pack-gqa-critical-path-2026` |
| FA4 Ragged TMA 描述符工作区 (`B=49, S=128, H=6`) | 5-D Wrap (~128KB 越界) + `300` 槽 (< `582` CTAs) | 每 CTA GMEM 描述符重写 + `888` 槽位上界 | **0 越界 Fault / 100% 覆盖 582 CTAs** | `src-fa4-ragged-kv-tma-per-cta-descriptor-2026` |
| veRL MoE LoRA 同步 (`16,000` 张量, 500 MiB, GB300) | 逐张量 2×设备克隆 + 逐张量 Pin (`705 ms`) | 单块 Pinned Host Staging + 零拷贝视图 (`131 ms`) | **5.38x 加速 (16,000 -> 1 次分配)** | `src-verl-vllm-lora-bucket-single-host-block-2026` |
| Mooncake EFA 滚动升级 (`120P + 120D` 节点) | 旧 Peer 泄漏（17 轮升级耗尽 4096 AH） | 重连自动驱逐过期 Peer 释放 AH | **活跃 AH 恒定 240** | `src-mooncake-efa-evict-stale-peer-reconnect-2026` |
