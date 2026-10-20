# NCCL Device API (`nccl_device.h`) & GIN 对称集合通信算例 / NCCL GIN Symmetric Collectives Simulation (`nccl_gin_symmetric_sim`)

## 中文概述 (Chinese Overview)

本实验基于 `NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`) 源码中的 `src/include/nccl_device/gin.h`、`src/device/symmetric/all_gather_gin.cuh`、`src/device/symmetric/reduce_scatter_gin.cuh` 与 `docs/examples/07_kernel_fusion/03_rmsnorm_gin/c/main.cu`，复现并验证：
1. **`rsAlgoHier` 两阶段 Warp Specialization 分区公式**：在 `512` 线程（`16` Warps，扣除 2 个 GIN 控制 Warp 后 `nWorkWarps = 14`）、`lsa.nRanks = 8`、`rail.nRanks = 8` 下，开启 `multimem=True` 时 `stage0_work = 7`、`stage1_work = 8`，工作 Warp 均衡切分为 `7 : 7`；关闭 `multimem=False` 时 `stage0_work = 56`、`stage1_work = 15`，重分配为 `11 : 3`。
2. **`ncclSymkRun_AllGather_RailRing_LsaSTMC` 格雷码（Gray-Code）地址打散**：`startOffsetMultiplier = lsaRank ^ ((lsaRank >> 1) & 1)` 生成 `[0, 1, 3, 2, 4, 5, 7, 6]` 8 个无冲突偏移，由 Warp 0 负责跨轨 `gin.put`、Warps `1..15` 并发执行机内 `bcastMultimem`。
3. **单算子融合 `RMSNormGIN`**：通过 `ncclGin_WeakSignalInc` 与 `gin.flush` 在单个 Kernel 内完成 `Reduce-Scatter + RMSNorm + All-Gather`，消除 `2` 次主机 Proxy 握手与中间 HBM 读写，HBM 流量降低 `20.0%`（`80.0 -> 64.0 MiB`），端到端延迟从 `234.57 us` 降至 `110.92 us`（**`2.11x`** 提速）。

## English Overview

This deterministic Python 3.12 standard-library experiment models `NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`):
1. **`rsAlgoHier` Two-Stage Warp Specialization (`reduce_scatter_gin.cuh`)**: For `512` CTA threads (`16` warps, `nWorkWarps = 14`) across `8x8` (`lsa=8, rail=8`), enabling NVLS `multimem=True` balances worker warps at **`7 : 7`**, whereas `multimem=False` shifts worker warps to **`11 : 3`** to absorb intra-node read amplification; inbox credits bound chunk sizes to **`[256 KiB, 2 MiB]`**.
2. **`AllGather_RailRing_LsaSTMC` Gray-Code Bank Diversification (`all_gather_gin.cuh`)**: `lsaRank ^ ((lsaRank >> 1) & 1)` produces the permutation `[0, 1, 3, 2, 4, 5, 7, 6]`, while Warp `0` drives inter-node `gin.put` and Warps `1..15` execute intra-node `bcastMultimem`.
3. **Single-Kernel `RMSNormGIN` Fusion (`03_rmsnorm_gin/c/main.cu`)**: Replaces 3 host-orchestrated kernels with a single device-initiated kernel using `ncclGin_WeakSignalInc`, cutting HBM traffic by **`20.0%`** (`80.0 -> 64.0 MiB`) and reducing latency from **`234.57 us`** to **`110.92 us`** (**`2.11x`** speedup).

## 运行与校验 / Run & Verify

```powershell
python experiments\nccl_gin_symmetric_sim\run.py --check
```
