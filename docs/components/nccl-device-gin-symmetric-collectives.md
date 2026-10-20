# NCCL Device API (`nccl_device.h`)、GPU 直发网络 (`ncclGin`) 对称集合通信与单算子 `RMSNormGIN` 融合架构

> **Language / 语言**: **中文 (Chinese)** | [English (英文版)](nccl-device-gin-symmetric-collectives.en.md)

## 1. 架构背景与核心痛点：为什么需要 `nccl_device` 与 `ncclGin` 对称集合通信？

在传统的宿主侧（Host-Initiated）NCCL 集合通信模式中，`ReduceScatter -> RMSNorm -> AllGather` 等张量并行（TP）/ 序列并行（SP）关键路径面临三重结构性税负：
1. **多算子发射与 Proxy 握手开销**：3 个独立 CUDA Kernel 的发射与 2 次跨流/Proxy FIFO 同步在短序列或小微批（Micro-batch）下消耗 **`41.5 us`** 固定延迟；
2. **中间激活的 HBM 往返流量（HBM Round-Trip）**：`ReduceScatter` 写回 HBM、`RMSNorm` 再次从 HBM 读取并写回、`AllGather` 再次从 HBM 读取，凭空增加 **`4 × shard_bytes`** 的额外 HBM 带宽占用；
3. **机内 NVLS `multimem` 与机间 Rail Ring RDMA 的流水线割裂**：缺少 CTA 内部 Warp 级别的细粒度协同机制。

基于 **`NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`)** 最新合入的 `src/include/nccl_device/gin.h`、`src/device/symmetric/all_gather_gin.cuh`、`src/device/symmetric/reduce_scatter_gin.cuh` 与 `docs/examples/07_kernel_fusion/03_rmsnorm_gin/c/main.cu` 一手源码，本文系统拆解 NCCL Device-Initiated Networking (`ncclGin`) 的完成语义、两阶段 Warp Specialization 对称集合通信及算子融合架构。

---

## 2. 一手源码机制拆解 (`implementation`)

### 2.1 `src/include/nccl_device/gin.h`：强/弱完成信号与物理内存段标签

在 `src/include/nccl_device/gin.h` (`blob 24ba3855188d15ac47c2b107552b2b31e3c1273c`) 中，`ncclGinSession::put` 定义了显式区分顺序保证的远端完成动作（Completion Actions）与本地计数器：

- **强信号（Strong Signal：`ncclGin_StrongSignalInc` / `ncclGin_StrongVASignalInc` / `ncclGin_StrongSignalAdd`）**：当远端对等体观测到该信号递增时，**保证此前所有（All preceding PUTs）发往该对等体的 PUT 均已落盘可见**；
- **弱信号（Weak Signal：`ncclGin_WeakSignalInc` / `ncclGin_WeakVASignalInc` / `ncclGin_WeakSignalAdd`）**：**仅保证与该信号绑定（Bundled）的当前单次 PUT 落盘可见**，允许网络多路径（Multi-Rail / Packet Spraying）下不同 PUT 之间免屏障乱序到达；
- **弱本地计数器（`ncclGin_WeakCounterInc`）与 `gin.flush(warps)`**：仅保证本地源缓冲区（Source Buffer）已被网卡 DMA 消费完毕，可安全复用或覆写；
- **物理 `cuMem` 段类型标签（`ncclGinSegmentType_t`）**：
  - `ncclGinSegmentTypeDevice = 0`（全设备 HBM 段）
  - `ncclGinSegmentTypeMixed = 1`（`HOST_NUMA` 与 Device HBM 混合段）
  - `ncclGinSegmentTypeHostNuma = 2`（全 CPU `HOST_NUMA` 段）。

### 2.2 `src/device/symmetric/all_gather_gin.cuh`：格雷码（Gray-Code）地址打散与单 Warp 跨轨 + 多 Warp `bcastMultimem`

在 `ncclSymkRun_AllGather_RailRing_LsaSTMC` (`blob 8a39c5a3228535b8e5e9954ebde83942d5ebab11`) 中，NCCL 将机间 Rail Ring 与机内 LSA（Load/Store Accessible）NVLS 多播融合在同一个 CTA 内：

```cpp
// Rotate by LSA rank with pattern 0 1 3 2 4 5 7 6 to diversify addresses.
uint32_t startOffsetMultiplier = lsaRank ^ ((lsaRank >> 1) & 1);
size_t startOffset =
  nElts > 0 ? ((size_t)startOffsetMultiplier * ncclSymkMcPerRankOffsetBytes % nElts) / chunkSize * chunkSize : 0;
```

1. **格雷码地址偏移打散**：对机内 8 个 `lsaRank`（`0..7`），二进制格雷码 `lsaRank ^ ((lsaRank >> 1) & 1)` 生成确定性置换序列 **`[0, 1, 3, 2, 4, 5, 7, 6]`**。这使机内 8 张 GPU 在同时发起跨节点 Ring 传输与 NVLS 多播写时，访问不同的 HBM Memory Controller / NVSwitch 地址段，消除同相位突发（Incast）竞争；
2. **非对称 Warp 分工**：
   - **`threadIdx.x < WARP_SIZE`（Warp 0，`ringThreads = 32`）**：独占负责跨节点 Rail Ring 的 `gin.put(rail, nextPeer, ..., ncclGin_SignalInc{railSignals + rail.rank})` 与 `gin.waitSignal`；
   - **`threadIdx.x >= WARP_SIZE`（Warps `1 .. blockDim.x/32 - 1`，共 15 个 Warp）**：等待 `railSignals + prevPeer` 到达后，立即调用 `bcastMultimem` 将跨轨收到的分片通过 NVSwitch 硬件多播广播至本机所有 LSA 对等卡。

### 2.3 `src/device/symmetric/reduce_scatter_gin.cuh`：`rsAlgoHier` 两阶段 Warp Specialization 与信用分块边界

在 `rsAlgoHier` (`blob 00ced10cbd1945817c743ec638a221b59ca7555d`) 中，每个 CTA 除 2 个 GIN 控制 Warp 外，其余 `nWorkWarps = blockDim.x / 32 - 2` 个工作 Warp 按两阶段（Stage 0 发送/预归约 vs Stage 1 跨轨接收/最终归约）的实际读取工作量动态切分：

$$\text{stage0\_work} = \begin{cases} 0, & \text{lsa.nRanks} = 1 \\ (1 \text{ if multimem else lsa.nRanks}) \times (\text{rail.nRanks} - 1), & \text{lsa.nRanks} > 1 \end{cases}$$

$$\text{stage1\_work} = (1 \text{ if multimem else lsa.nRanks}) + \text{rail.nRanks} - 1$$

$$\text{stage0\_nWorkWarps} = \min\left(\text{round}\left(\frac{\text{nWorkWarps} \times \text{stage0\_work}}{\text{stage0\_work} + \text{stage1\_work}}\right),\ \text{nWorkWarps} - 1\right)$$

同时，`ncclGinInboxA2ASession` 的流控信用（Credit Logic）对单块大小施加硬下界约束：

$$\text{minChunkBytes\_log2} = \text{ginInboxRail.size\_log2} - \min\left(\lfloor\log_2(\text{rail.nRanks}-1)\rfloor + \text{ncclGinScratchMaxBufsPerPeer\_log2},\ \text{ncclGinScratchMaxBufs\_log2}\right)$$

### 2.4 `docs/examples/07_kernel_fusion/03_rmsnorm_gin/c/main.cu`：单算子三阶段 `RMSNormGIN`

在 `RMSNormGIN` (`blob 7a50a31ff62f18a0ba1ef0b044314c210e0a2395`) 中，Grid 按 `tokens_per_gpu` 划分（每个 Token 一个 CTA），在**单次 Kernel Launch** 内顺序完成：
1. **Phase 1（Reduce-Scatter via `gin.put` + `ncclGin_WeakSignalInc`）**：各 CTA 将本卡负责 Token 的部分激活推送至目标 Rank 的 `window_recv`，通过 `gin.waitSignal` 等待 `nRanks` 个对等贡献到齐后在寄存器/共享内存 `reduction_buffer[]` 内求和；
2. **Phase 2（In-SM RMSNorm）**：直接在片上归约缓冲区计算均方根并归一化，**零中间 HBM 落盘**；
3. **Phase 3（All-Gather via `gin.put`）**：将归一化结果通过 `gin.put` 广播回各对等体的 `window_send`，信号值跨阶段单调累加（无需阶段间重置）。

---

## 3. 原创确定性离线实验 (`own-offline`) 与量化结论

通过 [`experiments/nccl_gin_symmetric_sim/run.py`](../../experiments/nccl_gin_symmetric_sim/run.py) 对 64 卡（`8 rail × 8 lsa`，`CTA = 512 threads = 16 warps`）及 8 卡 `T=4096, H=4096` `RMSNormGIN` 融合路径进行确定性仿真（见 [`results.json`](../../experiments/nccl_gin_symmetric_sim/results.json)）：

| 評估维度 | 传统三算子/无 Multimem 基线 | NCCL `ncclGin` 对称集合通信与融合路径 | 架构收益 / 不变量验证 |
|---|---|---|---|
| `rsAlgoHier` 工作 Warp 分配 (`14` work warps) | `multimem=False`：`stage0=11`, `stage1=3` (`56:15` 读放大) | `multimem=True`：**`stage0=7`, `stage1=7`** (`7:8` 均衡流水) | 硬件多播消除机内 `8x` 重复读，释放 4 个 Warp 给 Stage 1 跨轨归约 |
| `AllGather_RailRing_LsaSTMC` 地址偏移 | 朴素同偏移 `0`（8 卡并发打爆同一 MC 通道） | 格雷码乘子 **`[0, 1, 3, 2, 4, 5, 7, 6]`** | **8 个唯一偏移**（间隔 `256 KiB`），`1` GIN Warp + `15` Multimem Warps |
| `ncclGinInboxA2ASession` 分块窗口 | 任意小块触发每对等体信用溢出死锁 | **`[256 KiB, 2,097,152 B]`** (`4 MiB` Inbox) | 严格遵守 `MaxBufsPerPeer_log2=2` 与 `MaxBufs_log2=5` 信用守恒 |
| 分布式 `ReduceScatter + RMSNorm + AllGather` | 3 次发射 + 2 次 Proxy 握手：**`234.57 us`**，HBM **`80.0 MiB`** | 单算子 `RMSNormGIN` (`WeakSignalInc`)：**`110.92 us`**，HBM **`64.0 MiB`** | 端到端延迟提速 **`2.11x`**，发射开销降 **`6.38x`**，HBM 流量省 **`20.0%`** |

---

## 4. 解决方案架构师决策指南与可证伪问题

1. **何时启用 `RMSNormGIN` / Device-Initiated 融合算子？**
   - 当微批 Token 数满足 `tokens_per_gpu >= num_sms` 且单步通信载荷处于 `256 KiB – 32 MiB` 的延迟敏感区间时，单算子 `ncclGin` 融合可消除 `35 us` 级 Host Proxy 气泡与 `20%` HBM 往返；
2. **何时退回常规集合通信？**
   - 当未启用对称内存窗口（`ncclCommWindowRegister`）或网卡不支持 GDAKI / Proxy GPU Context 时，必须回退至宿主发起的 `ncclReduceScatter` + `ncclAllGather`。
3. **后续可证伪问题**：在跨机架 RoCEv2 Packet Spraying 乱序度超过 16 个包时，`ncclGin_WeakSignalInc` 与 `ncclGin_StrongSignalInc` 在 NIC 完成队列上的尾延迟差异是多少？
