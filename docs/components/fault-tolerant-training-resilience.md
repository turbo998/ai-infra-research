# 万卡同步训练容错与静默数据损坏（SDC）防御：`NVRx` 进程内热重启、对等内存 Checkpoint 与正交梯度校验和

## 1. 核心架构问题与结论摘要

当大模型同步预训练集群从 `4,096` 卡扩展至 `16,384`–`65,536` 张高功率液冷 GPU 时，即便单卡硬件与光模块的平均无故障时间（`MTBF`）高达 `12,000 小时`（约 `500 天`），整个集群的联合 `MTBF` 也会按 $\text{MTBF}_{\text{cluster}} = \text{MTBF}_{\text{GPU}} / N_{\text{GPU}}$ 呈反比坍塌：在 `16,384` 卡下每 **`43.95 分钟`** 发生一次故障，而在 `65,536` 卡下每 **`10.99 分钟`** 就会触发一次硬件/链路中断。

若沿用传统的 **Slurm 整作业销毁重排 + 并行文件系统（PFS）全量 Checkpoint 读写**（单次故障重启耗时 `915–960 s`，阻塞落盘 `95 s+`），万卡集群将陷入“重启尚未完成或刚跑数分钟即遭遇下一次故障”的雪崩死锁。本研究通过核验 `NVIDIA/nvidia-resiliency-ext`（`NVRx`）一手源码并构建离线恢复与 SDC 仿真模型（`experiments/megascale_fault_tolerance/`），确立以下三项核心工程结论：

1. **64K 卡集群下 `Slurm + PFS` 的有效训练时间比（`ETTR`）暴跌至 `18.58%`，而 `NVRx` 进程内热重启 + 对等节点内存恢复维持 `90.30%`（净增 `+71.72 percentage points`，有效算力提升 `4.86x`）**：
   - `NVRx` 的 `inprocess.Wrapper`（`wrap.py`）在不退出操作系统进程、不重新拉起 `srun`/`torchrun` 与 CUDA Runtime 上下文的前提下，通过 `AbortTorchDistributed` 中止挂起的 NCCL 通信域，激活预先驻留在 `reserve_fn` 循环中的热备节点（`RankAssignment` / `ShiftRanks`），将进程内通信域重构压缩至 **`18.0 s`**；配合 `LocalCheckpointManager` 与 `CliqueReplicationStrategy`（跨对等节点 CPU RAM / RDMA 环网拉取分片副本，`1.8 s`），单次故障总停机时间降至 **`19.8 s`**，异步快照阻塞开销仅 **`1.2 s`**。
2. **Young-Daly 最优 Checkpoint 周期随异步内存快照从 `11.8 分钟` 缩短至 `40–80 秒`**：
   - 依据一阶 Young-Daly 公式 $\tau^* = \sqrt{2 \, T_{\text{ckpt}} \cdot \text{MTBF}_{\text{cluster}}}$，在 `16,384` 卡下传统 PFS 方案最优周期高达 `707.8 s`（平均每次故障损失 `353.9 s` 已算训练步），而 `NVRx` 内存/NVMe 双层异步复制将最优周期压至 **`79.5 s`**（`65,536` 卡下为 **`40.0 s`**），将故障回滚重算损失削减 **`8.9x–17.7x`**。
3. **静默数据损坏（`SDC`）在 `BF16` 下 `100%` 逃逸常规 `NaN/Inf` 看门狗，正交 ABFT/梯度校验和以 `0.0732%` 算力开销实现单步 `100%` 拦截**：
   - 对典型有限 `BF16` 权重/梯度（如 `0.375`，`exp=125`）遍历全部 `16` 个比特位翻转发现，所有 `16` 种单比特翻转均产生有限浮点数（`exp < 255`，例如翻转第 `14` 位指数位使数值暴增 $2^{64}$ 倍至 `~1.5e18`，但在 `BF16` 中仍不是 `NaN` 或 `Inf`）。仅依赖 `isnan`/`isinf` 看门狗的即时检出率为 **`0.0%`**，平均需扩散 `~250` 步直至优化器动量溢出才被迫大回滚；而在 GEMM 与 AllReduce 边界引入行列正交 `FP32` 校验和（Orthogonal ABFT Checksum），仅需 **`0.0732%`** 附加浮点运算（`4096^3` 矩阵）即可在 **`1` 个微步内 `100%` 捕获并定位故障 GPU**。

---

## 2. 一手源码机制剖析：`NVIDIA/nvidia-resiliency-ext` (`NVRx`)

在 `NVIDIA/nvidia-resiliency-ext`（commit `b944b820bbea9e4732862fbf26046d2cc1b07877`）中，容错训练管线由三大子系统协同构成：

### 2.1 `inprocess.Wrapper` 进程内热重启与热备池调度

- **`src/nvidia_resiliency_ext/inprocess/wrap.py`（blob `b6d65bf6c117a59899d2650d979246268f5d2b0d`）与 `rank_assignment.py`（blob `1d2a0a261a985c430d166e9b88ffd0cc1905221d`）**：
  - `Wrapper` 装饰分布式 PyTorch 主训练函数。每个 Worker 进程同时启动后台监控线程 `MonitorThread`（`monitor_thread.py`）、独立守护进程 `MonitorProcess`（`monitor_process.py`）以及心跳看门狗 `ProgressWatchdog`（`progress_watchdog.py`）。
  - **热备节点池（`reserve_fn`）**：集群分配时预留少量 Inactive Ranks，它们不进入主训练步，而是阻塞在 `reserve_fn(state, store, progress_watchdog, progress_watchdog_interval)` 中定期向 `TCPStore`/`PrefixStore` 发送 `progress_watchdog.ping()` 并等待 `store.wait_for_completed()`。
  - **零进程销毁快速切换**：当任意活跃 Rank 触发 NCCL 通信超时、CUDA 异步异常或未通过链式硬件体检（`ChainedGPUHealthCheck`、`ChainedNicHealthCheck`、`ChainedNVLHealthCheck`）时，`MonitorThread` 抛出异步异常 `RankShouldRestart` 并调用 `AbortTorchDistributed` 立即终止底层 `c10d` / `ProcessGroupNCCL` 集合通信核，随后由 `RankAssignment`（如 `ActivateAllRanks` + `ShiftRanks`）剔除 `terminated_ranks`，将热备 Rank 编入新 `world_size` 并直接在**同一 OS 进程内**重新进入被包装的训练函数入口。

```
+----------------------------------------------------------------------------------------------------+
| NVRx In-Process Hot Restart (`inprocess.Wrapper`) vs Traditional Slurm/Torchrun Cold Restart       |
+----------------------------------------------------------------------------------------------------+
| [Traditional Slurm + PFS Restart: 915 - 960 s]                                                     |
|   Fault -> NCCL Timeout (300-600s) -> Kill all 64K Python procs -> Slurm epilog/prolog realloc     |
|         -> Cold Python import + CUDA init -> Read 10TB Checkpoint from congested Lustre/PFS        |
|                                                                                                    |
| [NVRx In-Process + Peer-RAM Replication: 19.8 s total]                                             |
|   Fault -> ProgressWatchdog / MonitorThread detects in <5s -> AbortTorchDistributed aborts NCCL    |
|         -> ChainedGPU/Nic/NVLHealthCheck isolates bad rank -> ShiftRanks activates hot reserve_fn  |
|         -> Re-init ProcessGroup in same OS process (18.0 s)                                        |
|         -> CliqueReplicationStrategy fetches shard from Peer CPU-RAM over RDMA (1.8 s) -> Resume!  |
+----------------------------------------------------------------------------------------------------+
```

### 2.2 `LocalCheckpointManager` 与 `CliqueReplicationStrategy` 内存/NVMe 副本

- **`src/nvidia_resiliency_ext/checkpointing/local/ckpt_managers/local_manager.py`（blob `fa37ebafdf790ae95b76e4f3591f6dec5f5a5464`）与 `replication/strategies.py`（blob `4e5a125a479098cc6b086d4deade1a930e7f9d73`）**：
  - 在 ZeRO / FSDP / Megatron 分布式优化器下，每个 Rank 仅持有自身负责的参数与优化器状态分片（`TensorAwareStateDict`）。`LocalCheckpointManager` 将快照写入本地 `/dev/shm`（RAM disk）或本地 NVMe（`root_local_ckpt_dir / session_id / rank`），将主训练流阻塞压缩至 Device-to-Host（D2H）Pinned Memory 拷贝耗时（`~1.2 s`）。
  - 为了防止单机掉电导致本地分片丢失，`CliqueReplicationStrategy` 在后台进程组（`local_group`，通常跨不同机柜/交换机叶节点的 `2` 或 `4` 个对等 Rank 组成 Clique）内通过 `replicate()` 交换分片副本；恢复时通过 `retrieve_plan(globally_available_ids, wanted)` 自动生成最小通信恢复计划 `ExchangePlan`，由存活对等节点直接通过 RDMA 将缺失分片推送给接替的热备 Rank。

---

## 3. 原创 CPU/Offline 万卡容错与 SDC 仿真实验分析

本仓在 `experiments/megascale_fault_tolerance/run.py` 中对 `4,096` 至 `65,536` 卡规模下的三种容错架构进行了端到端 Young-Daly + 泊松故障更新（Poisson Renewal）与慢节点（Straggler）耦合仿真。

### 3.1 `4,096–65,536` GPU 集群有效训练时间比（`ETTR`）对比矩阵

| 集群规模 (`n_gpus`) | 集群联合 `MTBF` (`分钟`) | Slurm + PFS 重启耗时 (`s`) | Slurm + PFS 最优快照间隔 (`s`) | Slurm + PFS `ETTR` (`%`) | TorchElastic + 本地 NVMe `ETTR` (`%`) | NVRx 进程内热备 + 对等内存恢复耗时 (`s`) | NVRx 最优快照间隔 (`s`) | **NVRx `ETTR` (`%`)** | **NVRx 相对 Slurm 净增益 (`pts`)** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`4,096`** | `175.78 min` | `915.0 s` | `1415.6 s` | `74.85%` | `90.07%` | `19.8 s` | `159.1 s` | **`96.97%`** | **`+22.12 pts`** |
| **`8,192`** | `87.89 min` | `915.0 s` | `1001.0 s` | `66.36%` | `87.59%` | `19.8 s` | `112.5 s` | **`96.19%`** | **`+29.83 pts`** |
| **`16,384`** | `43.95 min` | `915.0 s` | `707.8 s` | `54.86%` | `83.69%` | `19.8 s` | `79.5 s` | **`95.01%`** | **`+40.15 pts`** |
| **`32,768`** | `21.97 min` | `930.0 s` | `707.8 s` | `35.65%` | `75.97%` | `19.8 s` | `56.2 s` | **`93.17%`** | **`+57.52 pts`** |
| **`65,536`** | `10.99 min` | `960.0 s` | `707.8 s` | `18.58%` | `65.16%` | `19.8 s` | `40.0 s` | **`90.30%`** | **`+71.72 pts`** |

### 3.2 静默数据损坏（SDC）比特翻转分析与正交 ABFT 校验

在电压裕度收紧、高温高密运行的万卡 GPU 集群中，SRAM/寄存器偶发比特翻转（Silent Data Corruption）是比显式硬件宕机更危险的隐形杀手：

| 校验机制 | 单比特翻转即时检出率 (`16` bits of `BF16`) | 未检出时的平均污染扩散步数 (`Rollback Blast Radius`) | `4096 x 4096 x 4096` GEMM 附加计算开销 (`%`) | 架构优缺点与适用阶段 |
| --- | --- | --- | --- | --- |
| **仅依赖 `NaN/Inf` 梯度看门狗** | **`0.0%`**（`16/16` 单比特翻转仍为有限数） | `~250 steps`（直至 Loss 发散并污染近期 Checkpoint） | `<0.005%` | 无法拦截指数高位（如 bit 14）或尾数翻转导致的有限数值畸变 |
| **正交行列 ABFT + AllReduce 梯度校验和** | **`100.0%`**（阈值 $\epsilon = 10^{-5}$ 捕获全部有限偏移） | **`1 micro-step`**（当步丢弃并触发 `ChainedGPUHealthCheck` 隔离） | **`0.0732%`** | 利用 $C e = A (B e)$ 与 $e^T C = (e^T A) B$ 双重校验定位单卡 SDC |

---

## 4. 解决方案架构师设计指南与计算练习

### 4.1 万卡训练容错分级防御清单（Defense-in-Depth）

1. **毫秒–秒级（`0.1–20 s`）**：启用 `NVRx inprocess.Wrapper` + `ProgressWatchdog`，配置 `1%–1.5%` 预热 Standby Ranks（`reserve_fn`），发生单节点/单光模块故障时在 `19.8 s` 内进程内切流。
2. **分钟级（`40–120 s`）**：启用 `LocalCheckpointManager` + `CliqueReplicationStrategy(target_device="cpu")`，每 `40–80 s` 将分片快照写入本机 Pinned RAM / NVMe 并向不同机柜的对等节点复制 `1` 份冗余。
3. **小时级（`30–60 min`）**：由后台异步 I/O 进程池（`AsyncCallsQueue` + `FileSystemWriterAsync`）在不阻塞训练流的前提下，将经过 ABFT 校验的健康 Checkpoint 归档至持久化并行存储（3FS / Lustre）。

### 4.2 架构师动手计算练习

- **练习问题**：某 `32,768` 卡集群单卡 `MTBF = 12,000 h`，若采用异步本地 NVMe Checkpoint（阻塞时间 $T_{\text{ckpt}} = 8.0\text{ s}$），请计算集群 `MTBF`（分钟）以及 Young-Daly 最优 Checkpoint 周期 $\tau^*$（秒）。
- **参考解答**：
  1. $\text{MTBF}_{\text{cluster}} = \frac{12000 \times 3600}{32768} = 1318.36\text{ s} \approx 21.97\text{ min}$。
  2. $\tau^* = \sqrt{2 \times 8.0 \times 1318.36} = \sqrt{21093.75} \approx 145.2\text{ s}$。

---

## 5. 证据链与后续可证伪问题

- **证据链索引**：
  - `ai-infra-ft-nvrx-001`：`NVIDIA/nvidia-resiliency-ext` `inprocess/wrap.py` & `rank_assignment.py`（commit `b944b820bbea9e4732862fbf26046d2cc1b07877`）——核验进程内函数重启包装器、健康检查链与 `reserve_fn` 热备池机制。
  - `ai-infra-ft-nvrx-002`：`NVIDIA/nvidia-resiliency-ext` `local_manager.py` & `replication/strategies.py`（commit `b944b820bbea9e4732862fbf26046d2cc1b07877`）——核验本地 RAM/SSD 检查点与 `CliqueReplicationStrategy` 跨节点分片复制恢复。
  - `ai-infra-ft-nvrx-003`：本仓可复现离线算例 `experiments/megascale_fault_tolerance/run.py` 与 `results.json`（`own-offline`）。
- **后续可证伪问题**：当在 `65,536` 卡集群中完全不配置空闲热备节点（`0%` Standby Pool），而是通过 `RankAssignment` 动态缩减 Data Parallel 度（Elastic DP Shrink）或重配置 Pipeline Parallel 微批次时，通信图重建与 ZeRO-3 优化器状态重分片的延迟是否会超过 `45 s`？
