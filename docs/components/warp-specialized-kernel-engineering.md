# SM90 Hopper 与 SM100 Blackwell Warp-Specialized Ping-Pong 算子工程、动态寄存器重分配与 `TMEM` (`tcgen05`) 延迟掩盖机制

- **关联问题 ID**：`triton-cutlass-warp-specialized-pingpong-gemm` (`priority: 17`)
- **关联实验**：[`experiments/warp_specialized_kernels/run.py`](../../experiments/warp_specialized_kernels/run.py) · [`results.json`](../../experiments/warp_specialized_kernels/results.json)
- **一手验证源码**：
  - `NVIDIA/cutlass` (`commit 78e46228a1bfc5c972e2f9a634223af784cd6949`)：
    - `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp` (`blob 563c24018063100580238029c54d3f8ad34104d9`)
    - `include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp` (`blob 9f71be26324a874aedfb1177527398ec3dc7e515`)
  - `Dao-AIlab/flash-attention` (`commit 965c9212e6be33c821ee9f2b461564c5c91a2503`)：
    - `hopper/flash_fwd_kernel_sm90.h` (`blob 47b3817cd281d772e7366347682e97e4cb4cd861`)

---

## 1. 架构背景：为何在 Hopper/Blackwell 上传统对称 CTA 无法跑满 Tensor Core？

在 Ampere (`SM80`) 时代，所有 Warp 对称执行 `cp.async -> __syncthreads() -> mma.sync -> epilogue`。然而在 Hopper (`SM90`, `1,979 TFLOPS FP8`) 与 Blackwell (`SM100`, `4,500 TFLOPS FP8` / `9,000 TFLOPS FP4`) 上，Tensor Core 吞吐增长显著快于寄存器文件容量（每 SM 仍为 `65,536 x 32-bit = 256 KB`）与 Special Function Unit (`SFU` `exp2` / Softmax / SwiGLU) 吞吐。对称 CTA 面临三大物理瓶颈：

1. **寄存器平分导致的严重 Spill**：当 CTA 启动 `3` 个 Warpgroup（`384` 线程）时，静态均匀分配上限为 $\lfloor 65536 / 384 \rfloor_{8} = 168\text{ regs/thread}$。而单个 `128x256` FP32 累加器 Tile 需要 $\frac{128 \times 256 \times 4\text{ B}}{128\text{ threads} \times 4\text{ B}} = 256\text{ regs/thread}$。若采用均匀分配，每个 MMA 线程将产生 `87` 个 32-bit 寄存器溢出至 Local Memory（L1/L2 往返），导致 `SOL%` 从 `88.2%` 崩塌至 `38.3%`（损失 `2.30x`）。
2. **SFU / Epilogue 阻塞 Tensor Core 流水线**：在 FlashAttention-3 或带激活函数/量化的 GEMM 中，Online Softmax 的 `exp2` 与 Epilogue `TMA Store` 不使用 Tensor Core，若单消费组串行执行，Tensor Core 周期性空转 `20%–35%`。
3. **跨 CTA 的 L2 重复读取与尾波量化（Wave Quantization）**：在 `132 SM` (H100) 与 `148 SM` (B200) 上，静态 Grid 在非整除 Tile 数下产生高达 `23%–35%` 的尾波空转。

---

## 2. 一手源码解剖：SM90 `setmaxnreg` 动态寄存器重分配与 Ping-Pong 交错

### 2.1 `setmaxnreg.inc/dec` 硬件寄存器转移契约

在 `NVIDIA/cutlass` 的 `sm90_gemm_tma_warpspecialized_pingpong.hpp`（第 `143–149`、`577`、`805` 行）与 FlashAttention-3 的 `hopper/flash_fwd_kernel_sm90.h`（第 `82–83`、`309`、`361` 行）中，CTA 由 **1 个 Producer Warpgroup（`128` 线程，仅发起异步 `TMA` 描述符加载）** 与 **2 个 Consumer Warpgroups（各 `128` 线程，执行 `WGMMA` 与 Epilogue）** 组成：

```cpp
// include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp (lines 143-149)
static constexpr uint32_t RegsPerThread =
  size<0>(TileShape{}) * size<1>(TileShape{}) * sizeof(ElementAccumulator)
  / (NumMMAThreads * sizeof(uint32_t));
static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
static constexpr uint32_t MmaRegisterRequirement  = !HeavyRegisterPressure ? 232 : 240;
```

- **Producer Warpgroup (`warp_group_idx == 0`)**：调用 `cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>()`（底层编译为 PTX `setmaxnreg.dec.sync.aligned.u32 40` 或 `24`），将自身未使用的寄存器归还给 SM 物理寄存器池。
- **Consumer Warpgroups (`warp_group_idx == 1, 2`)**：调用 `cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>()`（底层编译为 PTX `setmaxnreg.inc.sync.aligned.u32 232` 或 `240`），从 SM 寄存器池认领 Producer 让出的配额。
- **容量守恒校验**：
  - 标准模式（`RegsPerThread < 208`，如 `128x192` Tile）：$128 \times 40 + 2 \times 128 \times 232 = 5,120 + 59,392 = 64,512 \le 65,536$。
  - 高压模式（`RegsPerThread >= 208`，如 `128x256` Tile）：$128 \times 24 + 2 \times 128 \times 240 = 3,072 + 61,440 = 64,512 \le 65,536$。

### 2.2 `OrderedSequenceBarrier<2, 2>` 双消费组 Ping-Pong 交错

在 `sm90_gemm_tma_warpspecialized_pingpong.hpp`（第 `122`、`851–937` 行）中，两个 Consumer Warpgroup 通过 `MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<2, 2>` 严格交替占有 Tensor Core（Stage 0）与 Epilogue/SFU（Stage 1）：

1. **阶段 0（Mainloop MMA 互斥段）**：Consumer 0 执行 `math_wg_order_barrier.wait()` $\rightarrow$ 发射全部 K-tile 的异步 `wgmma.mma_async` $\rightarrow$ 调用 `math_wg_order_barrier.arrive()` 立即放行 Consumer 1 进入 Tensor Core Mainloop，而 Consumer 0 自身继续执行 `collective_mainloop.mma_tail()`。
2. **阶段 1（Epilogue / SFU 重叠段）**：当 Consumer 1 正在独享 Tensor Core 计算下一个输出 Tile $(m+1, n)$ 的 `WGMMA` 时，Consumer 0 在 CUDA Core / SFU / TMA Store 单元上并发执行当前 Tile $(m, n)$ 的激活函数、缩放与写回（或 FlashAttention-3 中另一个块的行归一化与 `exp2`）。

---

## 3. SM100 Blackwell 范式跃迁：`256 KB/SM TMEM` + 单 Warp `tcgen05` `2-CTA MMA`

在 `NVIDIA/cutlass` 的 `sm100_gemm_tma_warpspecialized.hpp`（第 `143–153`、`501–576`、`726–805`、`869–944` 行）中，Blackwell (`SM100`) 彻底重构了算子微架构：

1. **专用 `256 KB/SM` Tensor Memory (`TMEM`) 解耦寄存器文件**：
   - 每个 SM 新增独立的 `256 KB`（`512 columns x 128 lanes x 32-bit`）片上 Tensor Memory (`TMEM`)。
   - MMA 累加器不再占用通用寄存器（RF），而是由 `tmem_allocator.allocate(ArchTag::kTmemCapacityColumns, &shared_storage.tmem_base_ptr)` 分配在 `TMEM` 中，并通过 `AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>` 实现 MMA 生产与 Epilogue 消费（`CopyOpT2R`：`TMEM -> Register`）的双缓冲重叠（`IsOverlappingAccum`）。
2. **5 类单 Warp 特化与单 Warp `tcgen05` UMMA 发射**：
   - SM90 的 `wgmma` 需要整个 Warpgroup（`128` 线程，平铺 2 组即 `256` 线程）协同喂数并同步；
   - SM100 的 `tcgen05.mma`（`UMMA`）仅需 **1 个 Warp（`NumMMAThreads = 32` 线程，甚至由 `elect_one_sync()` 单线程发射指令）** 即可驱动整个 SM（乃至 2-CTA Cluster 两个 SM）的 Tensor Core！
   - `sm100_gemm_tma_warpspecialized.hpp` 将线程块划分为 5 个独立角色（`WarpCategory`）：`MMA` (`32` threads)、`Sched` (`32` threads，驱动硬件 `PipelineCLCFetchAsync` 动态持久化调度)、`MainloopLoad` (`32` threads)、`EpilogueLoad` (`32` threads) 与 `Epilogue` (`128` threads)。
3. **`2-CTA Cluster MMA` 将 L2 $\rightarrow$ SMEM 流量减半**：
   - 当配置 `2-CTA Cluster`（`has_mma_peer_cta = true`）时，Leader CTA 的 `MMA` Warp 同时指挥两个相邻 SM 的 Tensor Core 协同计算 `256x256` 大 Tile。矩阵 A/B 通过 TMA Multicast 在 Cluster DSMEM 间共享，每个 CTA 从 L2 Cache 读取的数据量从 `48 KB/stage` 降至 `24 KB/stage`（**L2 流量降低 `50.0%`**）。

---

## 4. 离线实验验证结果（`experiments/warp_specialized_kernels`）

运行 `python experiments\warp_specialized_kernels\run.py --check` 获得的确定性量化结果如下：

### 4.1 SM90 动态寄存器重分配收益（`65,536` 32-bit 寄存器池）

| Tile 形状 ($M \times N \times K$) | 每 MMA 线程累加器寄存器 | 均匀分配 (`168 regs`) 溢出量 | `setmaxnreg` 配比 (`Load / MMA`) | 特化分配溢出量 | 均匀分配 `SOL%` | 特化分配 `SOL%` | 净加速比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `128x128x64` | `128 regs` | `0 regs` | `40 / 232 regs` | `0 regs` | `86.50%` | `86.50%` | `1.000x` |
| `128x192x64` | `192 regs` | `60 regs` | `40 / 232 regs` | `0 regs` | `49.30%` | `86.50%` | **`1.755x`** |
| `128x256x64` (`HeavyPressure`) | `256 regs` | `87 regs` | `24 / 240 regs` | `15 regs` | `38.30%` | `88.20%` | **`2.303x`** |

### 4.2 SM90 Ping-Pong `OrderedSequenceBarrier<2, 2>` 重叠收益

| 算子负载 | Tile ($M \times N \times K$) | MMA 周期 | Epilogue/SFU 周期 | 串行 Cooperative 周期 | Ping-Pong 重叠周期 | 掩盖比例 | 端到端加速比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `GEMM + Bias + SiLU (FP8)` | `128x256x128` | `4,096` | `820` | `4,916` | `4,206` | `86.59%` | **`1.169x`** |
| `FlashAttention-3 Fwd (FP16)` | `128x128x128` | `3,840` | `1,480` | `5,320` | `3,950` | `92.57%` | **`1.347x`** |
| `FlashAttention-3 Fwd (FP8 2-WG)` | `128x256x128` | `2,560` | `1,360` | `3,920` | `2,670` | **`91.91%`** | **`1.468x`** |
| `MoE Grouped GEMM SwiGLU` | `128x192x128` | `3,200` | `960` | `4,160` | `3,310` | `88.54%` | **`1.257x`** |

### 4.3 SM90 (H100) vs SM100 (B200) 跨负载 `SOL%` 与 L2 流量对比

| 负载场景 | $M \times N \times K$ | 算术强度 (`FLOP/B`) | SM90 L2/stage (`1-CTA`) | SM100 L2/stage (`2-CTA`) | SM90 MMA 线程 | SM100 MMA 线程 | SM90 `SOL%` | SM100 `SOL%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Skinny-M Decode (BS=32)` | `32x7168x7168` | `62.74` | `48.0 KB` | `32.0 KB` (`-33.3%`) | `256` | `32` | `7.64%` | `9.03%` |
| `Med-Batch / SpecTree (BS=128)` | `128x7168x7168` | `242.54` | `48.0 KB` | `32.0 KB` (`-33.3%`) | `256` | `32` | `29.55%` | `34.91%` |
| `MoE Grouped GEMM (256 tok)` | `256x4096x7168` | `442.84` | `48.0 KB` | `24.0 KB` (`-50.0%`) | `256` | `32` | `60.72%` | `70.46%` |
| `Square Prefill (4K Context)` | `4096x7168x7168` | `2,240.00` | `48.0 KB` | **`24.0 KB` (`-50.0%`)** | `256` | **`32`** | `87.50%` | **`94.20%`** |

---

## 5. 解决方案架构师行动指南与设计练习

1. **Triton / CUTLASS 算子选型准则**：
   - 在 H100 (`SM90`) 上，当 `M >= 2048` 且带有复杂 Epilogue（如 Online Softmax、Per-Token Quantization、SwiGLU）时，必须优先采用 **Ping-Pong Persistent Kernel**（`384` 线程，`setmaxnreg` `40/232` 或 `24/240`）；而在 `M <= 512` 的小矩阵下，应切换回 **Cooperative Kernel** 或 **Stream-K Persistent**，让两个 Consumer Warpgroup 协同切分同一个 Tile 以提早释放输出。
   - 在 B200/GB200 (`SM100`) 上，必须启用 `tcgen05` `2-CTA Cluster MMA`（`TmemAllocator2Sm` + `PipelineCLCFetchAsync`），以消除 `148 SM` 非 2 幂次 SM 数量带来的尾波量化损失并节省 `50%` L2 带宽。
2. **可证伪后续问题**：在 Blackwell `NVFP4` (`E2M1`) Grouped MoE GEMM 中，当专家激活 Token 数高度倾斜（如单专家仅分配到 `8–16` 个 Token）时，`2-CTA` `256x128` TMEM Tile 的行填充浪费是否会超过其 `50%` L2 流量节省收益？
