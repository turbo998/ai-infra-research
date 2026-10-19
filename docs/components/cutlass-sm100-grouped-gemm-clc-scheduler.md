# CUTLASS SM100 Ptr-Array / Grouped GEMM：异步 TensorMapUpdater 与 Cluster Launch Control (`CLC`) 动态组调度

> **关联研究问题**：`cutlass-sm100-grouped-gemm-clc-moe-scheduler`（Priority 26）
> **关联实验**：[`experiments/cutlass_sm100_grouped_clc_gemm/run.py`](../../experiments/cutlass_sm100_grouped_clc_gemm/run.py)
> **关联单测**：[`tests/test_cutlass_sm100_grouped_clc_gemm.py`](../../tests/test_cutlass_sm100_grouped_clc_gemm.py)
> **证据登记**：`ev-cutlass-sm100-array-tensormap-updater-2026`、`ev-cutlass-sm100-group-clc-scheduler-2026`、`ev-exp-cutlass-sm100-grouped-clc-2026`

---

## 1. 为什么 Blackwell SM100 Ptr-Array / Grouped GEMM 需要异步 `TensorMapUpdater` 与 `CLC`？

在稀疏 MoE（如 `DeepSeek-V3` 256 专家、`Qwen3-MoE` 128 专家）中，每个本地专家 $g \in [0, G-1]$ 接收到的 Token 数量 $M_g$ 高度偏斜（通常服从 Zipfian $s \approx 1.1\text{–}1.3$ 分布），且每个专家的激活/输出矩阵指针 $(A_g, B_g, C_g, D_g)$ 及 Leading Dimension $(lda_g, ldb_g, ldc_g, ldd_g)$ 在运行时动态变化。

传统 GPU Grouped GEMM 面临两大硬件瓶颈：
1. **同步 `Tensormap` 更新泡沫（Synchronous TMA Descriptor Update Stall）**：
   Hopper (`SM90`) 与 Blackwell (`SM100`) 的 Tensor Memory Accelerator (`TMA`) 要求每次跨组（`batch_idx` 切换）时在 Shared Memory 缓冲区中修改 128-Byte `CUtensorMap` 描述符（调用 `tensormap.replace.tile.global_address` / `global_dim` / `global_stride` PTX 指令），并执行 `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu` 栅栏将更新刷入全局 TMA 缓存。若该更新由 `Mainloop TMA Load` Warp 同步执行，每次切换专家组都会在关键数据加载路径上暴露约 `0.38–0.45 us` 的管线排空与栅栏停顿。当每 Cluster 需要跨越数十个细粒度专家时，TMA 更新停顿可占总 Kernel 时间的 `8%–14%`。
2. **静态轮询调度的尾波负载失衡（Static Round-Robin Wave Quantization Imbalance）**：
   若使用静态线性跨步（`current_work_linear_idx += gridDim`）分配各专家的 CTA Tile，由于不同专家的 `tensormap` 切换次数、Epilogue 尾块大小与 L2 Cache 命中率不同，各 2-CTA Cluster 的实际完成时间出现发散，导致最慢 Cluster 拖慢整个 Persistent Kernel（尾波失衡高达 `20%+`）。

---

## 2. 一手源码解剖：`NVIDIA/cutlass` SM100 异步 `TensorMapUpdater` 与 `Group` `CLC` 调度器

本研究直接核验了 `NVIDIA/cutlass`（commit `78e46228a1bfc5c972e2f9a634223af784cd6949`）中的三个核心头文件：
- `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp`（blob `a42c33c59f5fd4ab30a7bfe082d4a04de1e51fe9`）
- `include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp`（blob `b8daee03b47d80692d1ffa7898c927afd4595c93`）
- `include/cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp`（blob `e525cc5a59ec2115f7579763fb50d895ebea7cfd`）

### 2.1 双级 `clc_response` 与 `TensorMapReadyPipeline` 共享存储布局

在 `sm100_gemm_array_tma_warpspecialized.hpp`（lines 195–315）中，当 `IsTensorMapUpdateAsync == true`（针对输入维度动态变化的 Ptr-Array / Grouped GEMM）时，Kernel 将原本单级的 `clc_response` 扩展为两级流水线，并在 Shared Memory 中分配循环 `CUtensorMap` 槽位：

```cpp
// NVIDIA/cutlass: include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp (lines 199-243)
static constexpr bool IsTensorMapUpdateAsync =
    !(cute::is_same_v<decltype(ProblemShape{}), GroupProblemShape<Shape<int,int,int>>> &&
      IsInputTensorStatic);

using TensorMapReadyPipeline = cutlass::PipelineAsync<TensorMapPipelineStageCount>;
using CLCPipelineResponse = cute::conditional_t<
    IsTensorMapUpdateAsync,
    typename PersistentTileScheduler::CLCResponse::WithTensorMapUpdateInfo,
    typename PersistentTileScheduler::CLCResponse>;

struct PipelineStorageImplWithAsyncUpdate : PipelineStorageImpl {
  typename TensorMapReadyPipeline::SharedStorage tensormap_ready;
};
```

在 `SharedStorage` 中（lines 274–302）：
- `clc_response[2][SchedulerPipelineStageCount]`（32 B × 2 × 4 = `256 B`）：
  - `clc_response[0]`：由 Scheduler Warp 发出 `clusterlaunchcontrol.try_cancel` 硬件写入的原始 16-Byte CLC 响应槽位；
  - `clc_response[1]`：由 `tensor_map_updater` Warp 附加了 `WithTensorMapUpdateInfo`（包含 `uint16_t batch_changed` 与 `uint16_t TMA_stage`）的扩展响应槽位。
- `tensormaps[(NumTensorMapUpdaterThreads / NumThreadsPerWarp) + 1]`（128-Byte 对齐）：
  - 槽位 `0` 用于初始化拷贝；
  - 槽位 `1..N`（通常 `N=2` 个槽位，每个槽位含 Mainloop `A, B` 与 Epilogue `C, D` 共 4 个 128B `CUtensorMap`，合计 `1024 B`）供异步 Updater 循环填充。

### 2.2 独立 `tensor_map_updater` Warp 的非阻塞描述符更新路径

在 `sm100_gemm_array_tma_warpspecialized.hpp`（lines 1074–1276）中，CUTLASS 专门划分出一个 `tensor_map_updater` Warp（与 `scheduler`、`mainloop_load`、`mma`、`epilogue_load`、`epilogue` 正交并发）：

1. **寄存器主动释放**：调用 `cutlass::arch::warpgroup_reg_dealloc<GenericRegisterRequirement>()` 将寄存器缩减至 `32–40 regs/thread`，将寄存器文件空间让渡给 Epilogue 与 Mainloop Warpgroup。
2. **仅在 `did_batch_change == 1` 时触发 Tensormap 修改**：
   ```cpp
   // lines 1207-1255: tensor_map_updater loop
   uint32_t did_batch_change = prev_batch_idx != curr_batch_idx;
   if (did_batch_change) {
     collective_mainloop.tensormaps_perform_update<false /* WaitForInflightTmaRequests */>(
         shared_storage.tensormaps[updater_id + 1].mainloop,
         params.mainloop, work_tile_info.L_idx, problem_shape, ...);
     if (collective_epilogue.is_tensormap_update_required()) {
       collective_epilogue.tensormaps_perform_update<false>(...);
     }
   }
   ```
   注意模板参数 `WaitForInflightTmaRequests = false`：由于采用了多槽位循环 `tensormaps[updater_id + 1]` 并受 `tensor_map_ready_pipeline` 保护，Updater **无需等待当前正在飞行中的上一组 TMA 读取排空**，即可直接在下一循环槽位改写地址并提交 `tma_desc_commit_group()`。
3. **两阶段屏障交接**：Updater 更新完毕后写入 `clc_response[1][stage]` 并调用 `tensor_map_ready_pipeline.producer_commit()`；下游 `Mainloop Load` 与 `MMA` Warp 只需从 `clc_response[1]` 读取 `TMA_stage` 索引即可直接发射下一组专家的 `cp.async.bulk.tensor`，将 `94.5%` 的跨专家 TMA 描述符切换停顿隐藏在上一专家的 `tcgen05.mma` 尾声阶段。

### 2.3 `PersistentTileSchedulerSm100Group`：线性分簇扫描 + `CLC` 硬件工作窃取

在 `sm100_tile_scheduler_group.hpp`（lines 68–191）中，`advance_to_next_work` 结合了两种机制：
- **硬件 `CLC` 动态窃取**：当 `is_clc_query_enabled` 开启时，Scheduler Warp 调用 `clc_query_next_work_and_advance()`，向 Blackwell 硬件 Grid 调度单元发射 `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128`。硬件原子取消尚未启动的候选 Cluster，并将其平整线性 Tile ID 返回给当前空闲的 Persistent 2-CTA Cluster。
- **跨专家线性索引增量映射（`get_work_idx_m_and_n`）**：为了避免每次根据全局 `linear_idx` 从第 0 个专家重头累加求和，调度器维护了游标 `(group_idx, ProblemBlocks, CtaOffset)`，仅在 `linear_idx >= total_problem_blocks` 时向前推进 `group_idx++` 并跳过 $M_g = 0$ 的空专家（lines 147–185），使跨组寻址保持均摊 $O(1)$ 开销。

---

## 3. 定量基准实验与架构权衡（`experiments/cutlass_sm100_grouped_clc_gemm/`）

我们在 [`experiments/cutlass_sm100_grouped_clc_gemm/run.py`](../../experiments/cutlass_sm100_grouped_clc_gemm/run.py) 中构建了针对 **Blackwell B200**（`148 SMs = 74 2-CTA Clusters`，`2CTA Tile = 256×256×64`，`N=2048, K=2048`，`64` 本地专家，`8,192` 路由 Token，`Zipf(s=1.15)` 偏斜分布）的四模式对比仿真器。

### 3.1 四种 Grouped GEMM 调度架构端到端对比（64 专家，57 个非空组，540 个 2-CTA Tile）

| 调度架构模式 | 总耗时 (`us`) | 有效算力 (`TFLOPS`) | 暴露 TMA 更新停顿 (`us`) | 暴露 CLC 查询开销 (`us`) | Cluster 负载失衡度 (`(Max-Mean)/Mean`) | 相对 `sync_tma_static_group` 加速比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`host_launch_per_expert`**（逐专家 Host Kernel 启动） | `976.112` | `70.40` | `0.000` | `0.000` | `42.50%` | `0.147x` |
| **`sync_tma_static_group`**（静态轮询 + 同步 TMA 更新） | `143.083` | `480.28` | `12.541` (`8.76%`) | `0.000` | `21.04%` | `1.000x`（基线） |
| **`async_tma_static_group`**（静态轮询 + 异步 `TensorMapUpdater`） | `128.013` | `536.82` | `0.692` (`0.54%`) | `0.000` | `19.27%` | **`1.118x`** |
| **`async_tma_clc_dynamic`**（异步 `TensorMapUpdater` + `CLC` 动态窃取） | **`104.118`** | **`660.02`** | **`0.692`** (`0.66%`) | **`0.130`** (`0.12%`) | **`4.10%`** | **`1.374x`** |

### 3.2 专家数量扩展曲线（`G ∈ {16, 32, 64, 128}`，固定总 Token = `8,192`）

| 本地专家数 $G$ | 非空专家数 | 总 2-CTA Tiles | `sync_tma_static` (`us`) | `async_tma_static` (`us`) | `async_tma_clc_dynamic` (`us`) | 异步 TMA 提速 | `Async TMA + CLC` 总提速 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16** | `16` | `352` | `94.756` | `89.381` | **`70.517`** | `1.060x` | **`1.344x`** |
| **32** | `32` | `400` | `107.619` | `99.644` | **`79.640`** | `1.080x` | **`1.351x`** |
| **64** | `57` | `540` | `143.083` | `128.013` | **`104.118`** | `1.118x` | **`1.374x`** |
| **128** | `94` | `752` | `204.265` | `176.647` | **`141.732`** | `1.156x` | **`1.441x`** |

### 3.3 关键工程结论与适用边界

1. **专家越细粒度，异步 `TensorMapUpdater` 收益越高**：当本地专家数从 `16` 增至 `128` 时，平均每个专家的 M 维块数降至 `1–2` 个 Tile，几乎每次 Tile 推进都伴随 `did_batch_change == 1`。独立 `tensor_map_updater` Warp 将暴露 TMA 描述符更新停顿削减 **`94.5%`**（从 `12.54 us` 降至 `0.69 us`），单靠异步 Tensormap 更新即可带来 `1.12x–1.16x` 提速。
2. **何时禁用异步 `TensorMapUpdater`？**：如 `sm100_gemm_array_tma_warpspecialized.hpp` line 201 所示，当 `ProblemShape` 为静态均匀维度（`IsInputTensorStatic == true`，仅指针地址变化而张量维度/步长不变）或每个专家平均拥有 `> 16` 个 M-Tile 时，`did_batch_change` 触发频率极低，此时省下 `1,024 B` Shared Memory 与 32 个 Updater 线程更为划算。
3. **`CLC` 动态窃取消除 Zipfian 专家尾波**：在 `Zipf(1.15)` 偏斜分布下，静态步进会让部分 Cluster 连续撞上多次跨专家切换与尾块，造成 `21.04%` 的 Cluster 负载失衡；开启 `CLC` 硬件 `try_cancel` 后，先完成短 Tile 的 Cluster 动态接管尾部组 Tile，将失衡度压至 **`4.10%`**，实现 **`1.374x–1.441x`** 综合加速。
