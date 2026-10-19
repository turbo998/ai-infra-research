# PyTorch `CUDASymmetricMemory` 与 NVLS `multimem` 硬件多播集合通信架构

> **关联研究问题**：`nvshmem-ibgda-symmetric-heap-vmm-multicast-team`（Priority 26）
> **关联实验**：[`experiments/symm_mem_multimem_collective/run.py`](../../experiments/symm_mem_multimem_collective/run.py)
> **关联单测**：[`tests/test_symm_mem_multimem_collective.py`](../../tests/test_symm_mem_multimem_collective.py)
> **证据登记**：`ev-pytorch-symm-mem-multimem-ops-2026`、`ev-pytorch-symm-mem-inl-barrier-ptx-2026`、`ev-exp-symm-mem-multimem-collective-2026`

---

## 1. 为什么小消息低延迟推理/训练必须绕过传统 NCCL Ring/Tree？

在 `TP=8` 分布式推理 Decode 阶段（`BS=1..32`）以及细粒度 `Async-TP` 分块流水线中，每次 LayerNorm / 残差连接前后的 `AllReduce` 载荷往往只有 **`8 KB – 256 KB`**。
- 传统 `NCCL Ring/Tree` 需要通过主机侧 `ncclGroupStart/End` 打包、驱动队列下发、多跳 FIFO 缓冲区拷贝以及 `16–32` 个 GPU SM Block 的协议状态机同步，导致即使传输 `16 KB` 数据也存在 **`~8.5 us`** 的刚性延迟地板。
- PyTorch `torch.distributed._symmetric_memory`（`CUDASymmetricMemory`）利用 CUDA Virtual Memory Management (`cuMemCreate` / `cuMemMap` / `cuMemAddressReserve`) 与 NVLink Sharp (`NVLS` `cuMulticastCreate` / `cuMulticastBindMem`)，将节点内 8 张 GPU 的物理显存同时映射为：
  1. **Unicast P2P 虚拟地址数组** `void** buffer_ptrs[world_size]`（每张卡可直接对任意 Peer 发起 `ld.global` / `st.global`）；
  2. **NVLS 多播虚拟地址** `void* multicast_ptr`（单条 `multimem.ld_reduce` 由 NVLink Switch 在网内完成跨 8 卡归约，单条 `multimem.st` 由 Switch 硬件广播写回全体 GPU）。

---

## 2. 一手源码解剖：`pytorch/pytorch` `CUDASymmetricMemoryOps.cu` 与 `CUDASymmetricMemory-inl.cuh`

本研究直接核验了 `pytorch/pytorch`（commit `7f2c7e8ad5f8d42fc340abe7e1ee487afbfb9434`）中的两个核心实现文件：
- `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu`（blob `5be1174b1cca49fa3fadfe9491ae839f53dd4996`）
- `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh`（blob `e2475c36f125a6498e7a281a4f9bab29e3103331`）

### 2.1 无锁跨 GPU 系统级栅栏：`sync_remote_blocks<has_previous_mem_access, has_subsequent_mem_access>`

在 `CUDASymmetricMemory-inl.cuh`（lines 145–218）中，每个对称内存句柄关联一组跨卡映射的信号垫（`uint32_t** signal_pads`）。每个 CTA (`blockIdx.x`) 只与其余 Rank 上**相同 `blockIdx.x`** 的 CTA 进行点对点信号握手，从而避免全局跨块屏障：

```cpp
// pytorch/pytorch: torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh (lines 195-218)
template <bool has_previous_mem_access, bool has_subsequent_mem_access>
C10_DEVICE void sync_remote_blocks(
    uint32_t** signal_pads, size_t rank, size_t world_size) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    if constexpr (has_previous_mem_access) {
      // Wait for prior memory writes to become visible across NVLink fabric (.sys scope)
      atomic_cas_release_system(
          signal_pads[target_rank] + blockIdx.x * world_size + rank, 0, 1);
    } else {
      atomic_cas_relaxed_system(
          signal_pads[target_rank] + blockIdx.x * world_size + rank, 0, 1);
    }
    if constexpr (has_subsequent_mem_access) {
      spin_until_Acquire_system(
          signal_pads[rank] + blockIdx.x * world_size + target_rank, 1);
    } else {
      spin_until_relaxed_system(
          signal_pads[rank] + blockIdx.x * world_size + target_rank, 1);
    }
  }
  __syncthreads();
}
```
- **为什么首尾栅栏的模板参数非对称？**：
  - Kernel 入口栅栏调用 `sync_remote_blocks<false, true>`：在栅栏前本 Kernel 尚未写入对称内存（无需 `.release.sys`），但栅栏后紧接着要读取远端 Peer 缓冲区（必须用 `ld.global.acquire.sys` 确保前序算子写入全局可见）。
  - Kernel 出口栅栏调用 `sync_remote_blocks<true, false>`：栅栏前需用 `atom.global.release.sys.cas.b32` 刷出本 Kernel 的写入，而若后续无紧邻对称内存读取则省去 `.acquire.sys` 开销。

### 2.2 四种 SymmetricMemory AllReduce 内核的流量、SM 占用与数值精度对比

在 `CUDASymmetricMemoryOps.cu` 中，PyTorch 针对硬件是否支持 `multimem`（SM90 Hopper / SM100 Blackwell NVLS）与载荷大小提供了四种截然不同的实现：

| 内核函数 (`CUDASymmetricMemoryOps.cu`) | 硬件要求 | 最大 SM Blocks | 每 GPU NVLink 字节倍数 ($W=8$) | 累加精度与确定性语义 |
| :--- | :---: | :---: | :---: | :--- |
| **`one_shot_all_reduce_kernel`** (lines 524–607) | Unicast P2P | `24` (`512` threads) | $(W-1)B = \mathbf{7.0\times}$ | 各 Rank 按 `(rank+step)%W` 轮转发起 `ld_vec<16>` 防端口拥塞，并在寄存器中**固定按 `r = 0..W-1` 顺序**调用 `add_bf16x2` (`__hadd2`)，跨 Rank **100% 比特级一致**，但受限于 BF16 逐步舍入误差。 |
| **`two_shot_all_reduce_kernel`** (lines 610–754) | Unicast P2P | `24` (`1024` threads) | $2\frac{W-1}{W}B = \mathbf{1.75\times}$ | 阶段 1 各 Rank 负责 $1/W$ 切片的 P2P ReduceScatter；中间插入 `sync_remote_blocks<true, true>`；阶段 2 执行 P2P AllGather。跨 Rank **100% 比特级一致**。 |
| **`multimem_one_shot_reduce_kernel`** (`multimem_one_shot_all_reduce_out`, lines 183–246) | NVLS `multimem` | **`8`** (`1024` threads) | $\mathbf{1.0\times}$ Switch 归约读（本地 Unicast 写） | 每 Rank 对全量 $B$ 字节发出 `multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2` 并直接本地写 `output_ptr`（省去跨卡写回与出口屏障！）；网内 `acc::f32` 误差低，但不同 Rank 独立触发 Switch 归约时可能因端口到达序差异产生 **1-ULP 跨 Rank 漂移**。 |
| **`multimem_all_reduce_kernel`** (`multimem_all_reduce_`, lines 81–145) | NVLS `multimem` | **`8`** (`1024` threads) | $(1 + \frac{1}{W})B = \mathbf{1.125\times}$ | 每 Rank 仅对 $1/W$ 切片发出 `multimem.ld_reduce...acc::f32`（网内 FP32 累加归约），随后发出 `multimem.st.relaxed.sys.global.v4.f32` 硬件多播写回全体 Rank：既享受 **`acc::f32` 高精度**，又保证全体 Rank 接收同一份结果（**0% 跨 Rank 漂移**）。 |

---

## 3. 定量基准实验与架构选型（`experiments/symm_mem_multimem_collective/`）

我们在 [`experiments/symm_mem_multimem_collective/run.py`](../../experiments/symm_mem_multimem_collective/run.py) 中建模了 **8× H100 SXM5（NVLink 4.0，每方向 `450 GB/s`）** 从 `4 KB` 到 `64 MB` 的五种算法延迟曲线及 256 次随机张量数值舍入实验。

### 3.1 跨消息尺寸延迟对比（`W = 8` GPUs，单位：`us`）

| 消息尺寸 | `nccl_ring_standard` (`32` CTAs) | `symm_one_shot_p2p` (`<=24` CTAs) | `symm_two_shot_p2p` (`<=24` CTAs) | `symm_multimem_one_shot` (`<=8` CTAs) | `symm_multimem_two_shot_inplace` (`<=8` CTAs) | 最优算法与相对 NCCL 提速 |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`4 KB`** | `8.511` | `1.883` | `2.658` | **`1.869`** | `2.192` | `multimem_one_shot` (**`4.55x`**) |
| **`16 KB`** | `8.544` | `2.456` | `2.707` | **`2.187`** | `2.218` | `multimem_one_shot` (**`3.91x`**) |
| **`64 KB`** | `8.675` | `3.917` | `2.905` | `3.457` | **`2.321`** | `multimem_two_shot` (**`3.74x`**) |
| **`256 KB`** | `11.124` | `7.932` | `4.197` | `8.540` | **`2.874`** | `multimem_two_shot` (**`3.87x`**) |
| **`1 MB`** | `17.909` | `22.909` | `8.320` | `28.869` | **`4.812`** | `multimem_two_shot` (**`3.72x`**) |
| **`16 MB`** | `135.425` | `302.185` | `77.034` | `435.461` | **`47.281`** | `multimem_two_shot` (**`2.86x`**) |

### 3.2 关键发现：算法切换拐点与数值确定性权衡

1. **Unicast P2P 下 `One-Shot` vs `Two-Shot` 的精确拐点在 `~28 KB`**：
   - 当消息 `< 28 KB` 时，`one_shot_all_reduce` 省去了中间跨卡屏障（`0.78 us`），以 `1.88–2.46 us` 击败 `two_shot_all_reduce`；
   - 一旦消息 `>= 32 KB`，`one_shot_all_reduce` 的 $7.0\times$ Unicast P2P 读取流量成为瓶颈（在 `16 MB` 时耗时 `302.19 us`，比 `two_shot` 的 `77.03 us` 慢 `3.92x`）。
2. **`multimem` 仅用 `1/3` 的 SM（`8 blocks` vs `24–32 blocks`）即可跑满 NVLS 带宽**：
   - 因为归约计算与多播复制已卸载至 NVLink Switch ASIC，`multimem_all_reduce_kernel` 硬编码 `multimem_all_reduce_max_num_blocks = 8`，比传统 NCCL 释放了 `75%` 的通信 SM 占用，消除并发 GEMM 的 Wave Quantization 尾波膨胀。
3. **`acc::f32` 精度优势 vs `multimem_one_shot` 跨 Rank 1-ULP 风险**：
   - 在 256 组含通道离群值的 `W=8` 归约测试中，`multimem.ld_reduce...acc::f32.v4.bf16x2` 的平均绝对误差为 **`0.004056`**，比 P2P 顺序 `add_bf16x2` (`0.011068`) 降低 **`2.73x`**；
   - 但 `multimem_one_shot_all_reduce_out` 让每个 Rank 独立执行全量 `multimem.ld_reduce`，若 Switch 端口到达顺序发生抖动，会有 **`54.3%`** 的元素出现跨 Rank 1-ULP 差异；若上层算子（如 MoE Top-K 路由阈值或确定性采样）要求全 Rank 比特级严格一致，应优先选用 `multimem_all_reduce_`（由单一 Owner 归约后通过 `multimem.st` 广播，跨 Rank 漂移率 **`0.0%`**）。
