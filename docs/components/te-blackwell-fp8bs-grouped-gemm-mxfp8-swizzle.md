# TransformerEngine Blackwell FP8 Block-Scaling Grouped GEMM MXFP8 Swizzle 硬件仿真 (`grouped_swizzle_block_scaling_to_mxfp8_kernel`) 与 NVFP4 转置量化机制

> **English Version**: [te-blackwell-fp8bs-grouped-gemm-mxfp8-swizzle.en.md](te-blackwell-fp8bs-grouped-gemm-mxfp8-swizzle.en.md) | **关联实验**: [`experiments/te_grouped_mxfp8_swizzle_sim/README.md`](../../experiments/te_grouped_mxfp8_swizzle_sim/README.md)

### 一句话结论

由于 NVIDIA Blackwell（SM100+）上的 `cublasLt` 不再原生支持 Hopper 时代的 FP32 1D（`1x128`）与 2D（`128x128`）FP8 Block-Scaling GEMM，TransformerEngine（commit `f4a69dc81e`）通过 **零拷贝复用行主序 FP8 数据指针（`output->data.dptr == input->data.dptr`）**，并在持久化内核 `grouped_swizzle_block_scaling_to_mxfp8_kernel` 中将紧凑 FP32 块缩放因子（`scale_inv`）实时重排为 `128x4 = 512 B` 交错布局的 MXFP8 `E8M0` 缩放因子（每 `128x128` 数据 Tile 对应 `512 B` E8M0 Scale）；通过**设备端 32-lane `__shfl_up_sync` 排他前缀和建表（`tile_start, in_start, out_start`）+ 二分查找 `find_tensor`** 实现动态 Token 路由下的 CUDA Graph 安全重放，并在 256 专家 DeepSeek-V3 MoE 前向中将主机侧预分配上界控制在 `14,622,720 B`（相比精确需求 `11,296,768 B` 仅 `1.2944x` 冗余），同时在 `>2047` 张量时通过 `cudaFuncSetAttribute` 动态解锁超过 `48 KiB` 的共享内存上限（最高支持 `9,685` 个分组张量）。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `transformer_engine/common/include/transformer_engine/swizzle.h` | `NVIDIA/TransformerEngine` commit `f4a69dc81eabb5002a662e56a36e8454c9840a92` / blob `f85d88fdce83eb09253449fa3f98fa01e5522f51` | 声明 `nvte_swizzle_block_scaling_to_mxfp8_scaling_factors`、多张量批处理版以及 CUDA Graph 安全的分组版 `nvte_swizzle_grouped_block_scaling_to_mxfp8_scaling_factors` C ABI 契约。 |
| `transformer_engine/common/swizzle/swizzle_block_scaling.cu` | `NVIDIA/TransformerEngine` commit `f4a69dc81eabb5002a662e56a36e8454c9840a92` / blob `6cfa676592b02f21fefd9f60135ac5c9bd1b0e4f` | 实现 `grouped_swizzle_block_scaling_to_mxfp8_kernel`、二分查找 `find_tensor`、动态 SMEM 扩容 `enable_table_smem` 及主机侧上界 `grouped_mxfp8_scale_bytes_upper_bound`。 |
| `transformer_engine/pytorch/csrc/extensions/swizzle.cpp` | `NVIDIA/TransformerEngine` commit `f4a69dc81eabb5002a662e56a36e8454c9840a92` / blob `03eec7d80fd9c48b5cfa0758f17f470b62f7c422` | 实现 PyTorch 扩展层 `swizzle_scales_for_gemm` 与分组转换封装，管理 16B 对齐偏移与 `with_gemm_swizzled_scales` 状态位。 |
| `transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh` | `NVIDIA/TransformerEngine` commit `4a3a8a8e8578e96319467d0cc96b9f5db552a7d3` / blob `f83564b35fa5cb178f8bbd6333625d6e83842e73` | 实现 SM100 NVFP4 量化转置算子的 `compute_rowwise_amax_kernel` / `compute_columnwise_amax_kernel`（`ptx::abs_max_2x` 向量化归约）及 `wmma.load/store.m16n16k16.shared::cta` 共享内存内转置。 |

### 核心实现路径、数据布局与同步契约

1. **为什么 Blackwell 需要将 FP8 Block-Scaling 仿真为 MXFP8 Interleaved Layout（`swizzle.h:87-145`, `swizzle_block_scaling.cu:524-589`）**：
   - Hopper（SM90）在 `cublasLt` 中支持 `1x128`（1D）或 `128x128`（2D）FP32 `scale_inv` 的 FP8 块缩放矩阵乘；但在 Blackwell（SM100）上，第 5 代 Tensor Core（`tcgen05.mma`）硬件原生执行的是 **MXFP8 1D Scaling**（每 32 个 FP8 元素共享 1 个 8-bit `E8M0` 2 的整幂次指数 Scale，且要求按 `128x4` 交错 Tile 排布）。
   - 由于 `128` 恰好是 MXFP8 块大小 `32` 的 `4` 倍，每个 `128x128` 的 FP8 数据 Tile 在行方向包含 $128 / 32 = 4$ 个 MXFP8 缩放块、列方向包含 $128$ 行，因此其 MXFP8 `scale_inv` 恰好构成一个 `128 x 4 = 512 B` 的 `E8M0` 子块（`kTileOutputBytes = 512`）。只要原 FP8 Block-Scaling 的 `scale_inv` 已为 2 的幂次（或可无损抽取 IEEE-754 FP32 指数位转为 `E8M0`），即可在不觸碰底层 FP8 权重/激活数据（`output->data.dptr == input->data.dptr`）的前提下，将每个 `1x128`（1D）或 `128x128`（2D）FP32 Scale 复制并交错写入对应的 4 个 `E8M0` 槽位。
2. **CUDA Graph 安全的设备端 Warp 前缀和与二分查找 Tile 映射（`swizzle_block_scaling.cu:312-442`）**：
   - 在 MoE 动态路由（Grouped GEMM）中，每个专家分配到的 Token 数 `first_dims[t]`（或 wgrad 的 `last_dims[t]`）存放在 GPU 显存中，且每次 CUDA Graph Replay 都会改变。
   - 若在 Host 端读取维度并计算每个专家的 Tile 偏移，将引发 D2H 同步并彻底破坏 CUDA Graph 捕获。因此 `grouped_swizzle_block_scaling_to_mxfp8_kernel` 让每个 CTA 的 **Warp 0（`threadIdx.x < 32`）** 以 32 个张量为步长循环遍历全部 `num_tensors`，在每个步长内利用 5 轮蝶形洗牌 `__shfl_up_sync(0xFFFFFFFF, sum, d)`（$d \in \{1, 2, 4, 8, 16\}$）并行计算三个排他前缀和数组并写入动态共享内存：
     - `tile_start[t]`（长度 `num_tensors + 1`）：张量 $t$ 的起始 `128x128` Tile 序号（`tile_start[num_tensors]` 为总 Tile 数 `total_tiles`）；
     - `in_start[t]`（长度 `num_tensors`）：张量 $t$ 在紧凑 FP32 `scale_inv` 缓冲区中的起始 `float` 偏移；
     - `out_start[t]`（长度 `num_tensors`）：张量 $t$ 在交错 MXFP8 `E8M0` 缓冲区中的起始字节偏移。
   - 经 `__syncthreads()` 后，全网所有 Warp 以 `warp_stride = gridDim.x * WARPS_PER_TB` 步进遍历全局 `tile_id < total_tiles`，通过二分查找 `find_tensor(tile_start, num_tensors, tile_id)` 在 $O(\log_2 N)$ 步内定位所属专家 $t$，且天然跳过 0-Token 空专家（空专家的 `tile_start[t] == tile_start[t+1]`，永远不会成为二分查找的最终下界 `lo`）。
3. **动态共享内存 48 KiB 阈值与主机侧無同步上界公式（`swizzle_block_scaling.cu:358-360, 447-489`）**：
   - 每个 CTA 所需动态共享内存严格为 `table_smem_bytes(n) = (3 * n + 1) * sizeof(size_t) = 24n + 8` 字节。当 $n \le 2047$ 时（如 `2047 * 24 + 8 = 49,136 B <= 49,152 B`），直接运行在默认 `48 KiB` SMEM 下；一旦 $n \ge 2048$（如 `2048` 专家需要 `49,160 B`），`enable_table_smem` 通过 `std::call_once` 调用 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_bytes)`，在 Blackwell（`227 KiB = 232,448 B`）上最高支持 $n = 9,685$ 个分组张量。
   - 主机端在分配 `output->scale_inv` 时无需知道设备端每个专家的具体行数 $m_t$，而是直接利用逻辑总形状 $[F, L]$ 分配上界 `grouped_mxfp8_scale_bytes_upper_bound(t)`。

### 关键公式、单位与工程假设

1. **单张量输入 FP32 Scale 元素数与输出 MXFP8 E8M0 字节数**：
   $$\text{in\_floats}_{\text{1D}}(r, c) = \left\lceil \frac{c}{128} \right\rceil \cdot \operatorname{roundup}(r, 4), \qquad \text{in\_floats}_{\text{2D}}(r, c) = \left\lceil \frac{r}{128} \right\rceil \cdot \operatorname{roundup}\!\left(\left\lceil \frac{c}{128} \right\rceil, 4\right)$$
   $$\text{out\_bytes}_{\text{MXFP8}}(r, c) = \operatorname{roundup}(r, 128) \cdot \left\lceil \frac{c}{128} \right\rceil \cdot 4 = \text{tiles}(r, c) \times 512\text{ B}$$
2. **主机侧无 D2H 同步的 Grouped MXFP8 Scale 字节上界（`swizzle_block_scaling.cu:466-489`）**：
   - **均匀切分（Uniform）**：$B_{\text{uniform}} = n \cdot \operatorname{roundup}(F / n, 128) \cdot \lceil L / 128 \rceil \cdot 4$（精确值）
   - **首维可变（Varying First Dim，$\sum_{t=0}^{n-1} r_t = F$）**：每个张量最多因 128 行对齐多填充 127 行，故：
     $$B_{\text{varying\_first}}(F, L, n) = (F + 127n) \cdot \left\lceil \frac{L}{128} \right\rceil \cdot 4$$
   - **末维可变（Varying Last Dim，$\sum_{t=0}^{n-1} c_t = L$）**：每个张量在列方向最多多出 1 个未满 128 列的 Tile，故：
     $$B_{\text{varying\_last}}(F, L, n) = \operatorname{roundup}(F, 128) \cdot 4 \cdot \left(\left\lceil \frac{L}{128} \right\rceil + n\right)$$

### 合成实验与定量验证（`experiments/te_grouped_mxfp8_swizzle_sim`）

通过运行 `python experiments/te_grouped_mxfp8_swizzle_sim/run.py --check` 对 4 类典型 Blackwell 分组 GEMM 场景进行精确仿真（数据源自 `experiments/te_grouped_mxfp8_swizzle_sim/results/summary.json`）：

| 场景 (`scenario`) | 专家/张量数 $n$ | 逻辑形状 $[F, L]$ | `128x128` Tile 总数 | 动态 SMEM (`table_smem_bytes`) | 输入 FP32 Scale 字节 | 精确输出 MXFP8 Scale 字节 | 主机上界 (`host_upper_bound_bytes`) | 上界冗余比 (`slack`) | 输出/输入体积比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ds_v3_256e_1d_fprop`（含 16 空专家） | `256` | `[32768, 7168]` | `22,064` | `6,152 B` (`optin=false`) | `7,420,672 B` | `11,296,768 B` | `14,622,720 B` | `1.2944x` | `1.5223x` |
| `ds_v3_256e_2d_fprop`（2D 128x128 块） | `256` | `[32768, 7168]` | `22,064` | `6,152 B` (`optin=false`) | `88,256 B` | `11,296,768 B` | `14,622,720 B` | `1.2944x` | **`128.0x`** |
| `ultra_moe_2048e_optin_smem` | `2048` | `[65536, 4096]` | `65,536` | **`49,160 B` (`optin=true`)** | `8,388,608 B` | `33,554,432 B` | `41,680,896 B` | `1.2422x` | `4.0x` |
| `ds_v3_64e_wgrad_varying_k` | `64` | `[7168, 32768]` | `16,072` | `1,544 B` (`optin=false`) | `8,228,864 B` | `8,228,864 B` | `9,175,040 B` | `1.1150x` | `1.0x` |

### 架构师设计练习与可证伪问题

1. **为什么 2D FP8 Block-Scaling（`128x128` 块）在 Blackwell 上仿真为 MXFP8 时，Scale 显存体积会膨胀 `128.0x`（从 `88,256 B` 增至 `11,296,768 B`）？**
   - **解答**：2D Block-Scaling 中每个 `128x128` 数据 Tile 仅存储 1 个 `FP32`（`4 B`）缩放因子；而 Blackwell 的 `tcgen05` MXFP8 硬件指令要求按 `1x32` 粒度提供 `E8M0` 缩放因子，即每个 `128x128` 数据 Tile 必须填充 `128 行 x 4 列 = 512 B` 的 `E8M0` 字节（将同一个 `E8M0` 值在 128 行 × 4 列内复制 512 次），因此 `512 B / 4 B = 128.0x`。尽管 Scale 膨胀了 128 倍，但由于 FP8 主数据本身为 `32768 * 7168 = 234,881,024 B`（`224 MiB`）且零拷贝共享，额外 `11.3 MB` 的 Scale 仅占主数据的 `4.8%`，远优于将权重退回 BF16。
2. **可证伪假设**：若分组张量同时具有可变的 `first_dims` 和可变的 `last_dims`（即 `t.first_dims.has_data() && t.last_dims.has_data()`），`grouped_mxfp8_scale_bytes_upper_bound` 能否在不读取设备端数组的情况下给出紧凑线性上界？
   - **验证方式**：源码 `swizzle_block_scaling.cu:473-475` 显式断言 `NVTE_CHECK(!(t.first_dims.has_data() && t.last_dims.has_data()))` 拒绝该情况，因为当 $\sum r_t = F$ 且 $\sum c_t = L$ 独立变化时，$\sum \lceil r_t/128 \rceil \lceil c_t/128 \rceil$ 的最坏情况上界是乘积级而非紧致线性函数。
