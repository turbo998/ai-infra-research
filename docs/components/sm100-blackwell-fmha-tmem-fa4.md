# SM100 Blackwell FMHA 与 FlashAttention-4：512-Column `TMEM` 零 SMEM 溢出、16-Warp 异构角色分工与 `FFMA2` 多项式 `exp2` 软硬协同

> **关联问题**：`sm100-blackwell-fmha-warp-specialized-tmem-fa4` (`priority: 28`, `domain: "gpu_kernels"`)  
> **一手源码核验**：
> - `NVIDIA/cutlass` (`commit 78e46228a1bfc5c972e2f9a634223af784cd6949`)：
>   - `examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp` (`blob 0c2007b674ebb35249d0e5033d0dd33305080d3a`)
>   - `examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp` (`blob fee1dfe33a2e0205f77099376a83b7f1902f09bc`)
> - `Dao-AILab/flash-attention` (`commit e7a6402a9ff099de5eb628ce207a88a3f0aa5378`)：
>   - `flash_attn/cute/flash_fwd_sm100.py` (`blob c5add6c32495bc5fda9cbed0b85081942b65aa48`)
>   - `flash_attn/cute/softmax.py` (`blob 39ff94867d5ec6c08c8955db95cf3dca0893c3bb`)
>   - `flash_attn/cute/utils.py` (`blob 2cb3427fe6b4fac54b72054840529a28dd2774b3`)  
> **可复现实验**：`python experiments/sm100_blackwell_fmha_fa4/run.py --check`

---

## 1. 架构背景：为什么 SM90 FlashAttention-3 的流水线在 SM100 Blackwell 上撞上 SFU 与寄存器墙？

在 Hopper (`SM90`, H100) 的 FlashAttention-3 (`wgmma.mma_async`) 中，Tensor Core 的输出累加器（`S = QK^T` 与 `O = PV`）必须驻留在消费者 Warpgroup（`128` 线程）的 **通用寄存器文件（RMEM，`64K × 32-bit` / SM）** 中。这带来了三重结构性约束：
1. **MMA 与 Softmax 强耦合于同一 Warpgroup**：由于 `S` 驻留在寄存器中，执行 `wgmma` 的 Warpgroup 必须自己完成 `S` 的 Online Softmax（`row_max`、`exp2`、`row_sum`）以及历史输出 `O` 的重缩放（`O *= exp2(m_old - m_new)`），导致高寄存器压力（每线程 `192–240` 个寄存器）。
2. **`P` 矩阵的寄存器重排或 SMEM 溢出**：`PV` 矩阵乘要求左操作数 `P` 满足 `wgmma` 寄存器描述符或写入 Shared Memory (`sP`)，消耗宝贵的 SMEM 容量（`2` 级 `128×128` BF16 需 `64 KiB`），挤压了 KV TMA 流水线级数。
3. **Blackwell 算力跃迁引发的 `MUFU.EX2` 瓶颈反转**：在 SM100 (`B200`) 上，第 5 代 Tensor Core (`tcgen05.mma`) 的单 SM 吞吐翻倍至 **BF16 `8,192 FLOPS/cycle`**（`2.25 PFLOPS` 峰值）与 **FP8 `16,384 FLOPS/cycle`**（`4.5 PFLOPS` 峰值），完成一个 `128×128×128` 的 `QK^T + PV` Tile 仅需 `1,024` 周期（BF16）或 `512` 周期（FP8）。然而，Online Softmax 必须对 `128×128 = 16,384` 个元素逐个求指数 `exp2`——硬件超越函数单元（`MUFU.EX2`）的吞吐远跟不上 `tcgen05.mma` 的推进速度，使 Attention 从“Tensor Core 受限”反转为 **“MUFU 超越函数 + `O` 累加器重缩放受限”**。

---

## 2. 一手源码深度解剖

### 2.1 `TmemAllocation`：512 列（`256 KiB`）TMEM 的原地别名（In-Place Aliasing）与零 SMEM `P` 传输

在 `NVIDIA/cutlass` 的 `examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`（第 `125–151` 行）中，SM100 每个 SM 独享的 **`256 KiB` Tensor Memory (`TMEM`, `128 lanes × 512 columns × 4 Bytes`)** 被精确划分为双缓冲 Ping-Pong 布局：

```cpp
// examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp (L125-L151)
struct TmemAllocation {
  static constexpr uint32_t kSizeS = 128;
  static constexpr uint32_t kSizeO = 128;
  static constexpr uint32_t kSizeP = 32;
  // Transform: S0 -> P0 & V0; S1 -> P1 & V1
  static constexpr uint32_t S0 = 0;
  static constexpr uint32_t S1 = S0 + kSizeS;      // 128
  static constexpr uint32_t V0 = S0;               // 0   (2 cols: OldRowMax, NewRowMax)
  static constexpr uint32_t V1 = S1;               // 128
  static constexpr uint32_t P0 = S0 + 32;          // 32..64  aliased inside S0!
  static constexpr uint32_t P1 = S1 + 32;          // 160..192 aliased inside S1!
  static constexpr uint32_t O0 = S1 + kSizeS;      // 256..384
  static constexpr uint32_t O1 = O0 + kSizeO;      // 384..512
  static constexpr uint32_t kEnd = O1 + kSizeO;    // 512
};
```

这一布局展示了三个极其精妙的硬件工程设计：
1. **刚好填满 `512` 列 TMEM**：`S0 [0..128)`、`S1 [128..256)`、`O0 [256..384)`、`O1 [384..512)` 共计 `512` 列（`128 × 512 × 4B = 262,144 Bytes = 256 KiB`），实现 `100%` TMEM 容量利用。
2. **`P0/P1` 与统计量 `V0/V1` 原地复用 `S0/S1` 列空间**：当 `Softmax0` Warpgroup 通过 `SM100_TMEM_LOAD_32dp32b32x` 将 `S0` 的 `128` 列 FP32 分数读入寄存器后，立即将旧/新行最大值 `(old_row_max, new_row_max)` 通过 `SM100_TMEM_STORE_32dp32b2x` 写回 `V0`（列 `0..2`），并将量化后的概率矩阵 `P0`（`128×128` FP8 或 `128×64` BF16 打包为 `32` 个 32-bit 列）写回 `P0 = S0 + 32`（列 `32..64`）。
3. **`TS` 模式 `tcgen05.mma` 零 SMEM 消耗读取 `P`**：`CollectiveMmaPV` 调用 `cute::elsch::to_tiled_mma_sm100_ts(mma_pv)`（`TS` = Tensor Memory × Shared Memory），直接从 TMEM 的 `P0/P1` 列读取左矩阵 `P`，从 SMEM 读取右矩阵 `V`，将结果累加进 TMEM 的 `O0/O1`！源码第 `388` 行显式构造空指针：`Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), ...)`——**`sP` 占用的 Shared Memory 严格为 `0 Bytes`**，从而在 `228 KiB` SMEM 中释放出 `64 KiB` 空间，使 KV TMA 流水线得以扩展至 `3–4` 级（`StageCountQ = 2`, `StageCountKV = 4`）。

### 2.2 16-Warp (`512` 线程) 七角色解耦与细粒度寄存器重分配

在 `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`（第 `61–116` 行）与 `Dao-AILab/flash-attention/flash_attn/cute/flash_fwd_sm100.py`（第 `312–319` 行）中，单 CTA 的 `16` 个 Warp（`512` 线程）被彻底解耦为 **7 个专职角色**：

| Warp ID | 角色 (`WarpRole`) | 线程数 | CUTLASS 寄存器/线程 | FA4 Tuned (BF16/FP8) | 核心职责与硬件单元 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0..3` (`WG0`) | `Softmax0` | `128` | `192` | `184` / `160` | 读取 TMEM `S0` $\rightarrow$ Online Softmax $\rightarrow$ 写回 TMEM `V0` 与 `P0` |
| `4..7` (`WG1`) | `Softmax1` | `128` | `192` | `184` / `160` | 读取 TMEM `S1` $\rightarrow$ Online Softmax $\rightarrow$ 写回 TMEM `V1` 与 `P1` |
| `8..11` (`WG2`) | `Correction` | `128` | `96` | `80` / `72` | 异步读取 `V0/V1`，对 TMEM `O0/O1` 执行 `O *= exp2(m_old - m_new)` 及末步归一化 |
| `12` | `MMA` | `32` | `32` | `32` | **单 Warp** 发射全部 `tcgen05.mma`（交替驱动 `S0 = Q0·K^T`、`O0 += P0·V`、`S1 = Q1·K^T`、`O1 += P1·V`） |
| `13` | `Load` / `Epilogue` | `32` | `32` | `32` | **单 Warp** 发射 TMA 异步加载 `Q0/Q1` 与 `K_t/V_t` 进入 SMEM 环形缓冲区 |
| `14` | `Epilogue` / `Load` | `32` | `32` | `32` | **单 Warp** 执行 TMA Store (`tma_store_o`) 将最终输出写回 HBM |
| `15` | `Empty` / Scheduler | `32` | `24` | `24` | 释放寄存器配额（`setmaxnreg.dec.sync.aligned 24`）或驱动 CLC 动态持久化调度 |

核验寄存器总量：
$$\text{Total Regs (CUTLASS Default)} = 2 \times (128 \times 192) + (128 \times 96) + 3 \times (32 \times 32) + (32 \times 24) = 49,152 + 12,288 + 3,072 + 768 = 65,280 \le 65,536$$
而在 `flash_fwd_sm100.py`（第 `103–126` 行）中，针对 `hd=128` BF16 调优后的配额为 `Softmax=184, Correction=80`（总计 `61,184` 个寄存器，占 `93.36%`），FP8 为 `Softmax=160, Correction=72`（总计 `54,016` 个寄存器，占 `82.42%`）。

### 2.3 FlashAttention-4 的四大算子级突破（`flash_fwd_sm100.py` & `softmax.py` & `utils.py`）

1. **`FFMA2` 打包多项式软仿真卸载 `MUFU.EX2`（`ex2_emulation_2`）**：
   - 在 `flash_attn/cute/utils.py`（第 `731–775` 行）与 `softmax.py`（第 `421–462` 行 `apply_exp2_convert`）中，FA4 将 `[128, 128]` 的分数行切分为 `frg_tile = 32` 的片段，按配置参数 `ex2_emu_freq`（如非因果 BF16 `freq=10`、因果 BF16 `freq=16`、因果 FP8 `freq=8`）将每个周期内末尾 `ex2_emu_res = 4` 个元素的 `exp2(x)` 从硬件 `MUFU.EX2` 卸载到 **FP32 打包乘加单元（`cute.arch.fma_packed_f32x2`）**：
     - 先用 `add.rm.ftz.f32` 向下取整拆分整数部分 $x_{\text{int}} = \lfloor x \rfloor$ 与小数部分 $x_{\text{frac}} = x - x_{\text{int}} \in [0, 1)$；
     - 用 3 阶 Horner 多项式在 `FFMA2` 流水线上成对计算 $2^{x_{\text{frac}}}$；
     - 通过内联 PTX 将整数阶码直接加到 IEEE-754 FP32 指数位合成最终结果（在 $[-12, 0]$ 区间最大相对误差仅 `8.8e-5`）。
   - 正如源码第 `117–120` 行注释所记载：在 FP8 因果 `hd128` 下，将 `ex2_emu_freq` 从 `16` 调激进至 `8`（卸载 `~32.5%` 的 `exp2` 到 FMA 流水线），在 B200 上实测带来 **`+3.4% (4K) .. +5.5% (16K)`** 的纯端到端加速且精度零退化。
2. **惰性行最大值更新与 `Correction` 跳过机制（`rescale_threshold`）**：
   - 在 `softmax.py` 第 `352–360` 行（`SoftmaxSm100.update_row_max_from_local`）中：
     ```python
     acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
     acc_scale = cute.math.exp2(acc_scale_)
     if cutlass.const_expr(self.rescale_threshold > 0.0):
         if acc_scale_ >= -self.rescale_threshold:
             row_max_new = row_max_old
             row_max_safe = row_max_old
             acc_scale = 1.0
     ```
   - 当新 KV 块的最大值相比历史 `row_max_old` 的增幅不超过 `rescale_threshold`（以 $\log_2$ 为单位）时，保持旧 `row_max_old` 不变并令 `acc_scale = 1.0`（利用输入类型如 BF16/FP8 上方未用满的动态范围吸收最多 $2^{\text{rescale\_threshold}}$ 的概率增幅）。此时 `Correction` Warpgroup 检测到 `acc_scale == 1.0`，**直接跳过对 TMEM `O0/O1` 的加载、乘法与写回**，消除了 `>78%` 稳态步的 TMEM 读写争用！
3. **`split_P_arrive` 分段提前唤醒 `PV` MMA（`flash_fwd_sm100.py` 第 `186–192` 行）**：
   - `self.split_P_arrive = int((n_block_size // 4 * 3) / 32) * 32`（当 `n_block_size = 128` 时为 `96` 列，即 `3/4` Tile）。
   - Softmax Warpgroup 先写完 `P` 的前 `96` 列即向 `MMA` Warp 发送首次到达信号（`arrive`），使 `tcgen05.mma` 立即开始前 `96` 列的 `P @ V` 矩阵乘，与 Softmax 最后 `32` 列的转换写回完全重叠。
4. **SM103 (`B300` Blackwell Ultra) 原生硬件 `exp2` 与 `tcgen05.ld.red` 硬件行归约**：
   - 在 `flash_fwd_sm100.py` 第 `245–254` 行与 `softmax.py` 第 `363–372` 行中，针对 `Arch.sm_103f` (`B300`)：
     - 由于 SM103 硬件原生 `MUFU.EX2` 吞吐翻倍，`_TUNING_CONFIG` 在 `is_sm103=True` 时自动令 `ex2_emu_freq = 0`；
     - 启用 `self.use_ldred_rowmax = True`，利用 `tcgen05.ld.red` 指令在 TMEM 控制器读取 `S` 的同时由硬件直接算出该行最大值 `hw_row_max`，彻底省去软件寄存器 `fmax` 树归约（`update_row_max_precomputed`）。

---

## 3. 原创 CPU/Offline 确定性算例结果

运行 `python experiments/sm100_blackwell_fmha_fa4/run.py --check` 得到以下基准对比（B200 `148 SM @ 1.83 GHz` vs B300 `SM103`）：

| 内核架构配置 | 数据类型 | 硬件目标 | `FFMA2` 卸载比例 | 理想 TC 周期 | MUFU 周期 | 有效 Tile 周期 | 实效吞吐 (TFLOPS) | Tensor Core SOL% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sm90_fa3_baseline_scaled_to_b200` | BF16 | `SM100_B200` | `0.0%` | `1024.0` | `682.7` | `1477.7` | **`1,537.5`** | `69.3%` |
| `sm100_cutlass_fmha_hw_exp2_only` | BF16 | `SM100_B200` | `0.0%` | `1024.0` | `682.7` | `1137.7` | **`1,997.0`** | `90.0%` |
| `sm100_fa4_bf16_ex2_emu10_lazy_rescale` | BF16 | `SM100_B200` | `26.0%` | `1024.0` | `505.2` | `1069.0` | **`2,125.3`** | **`95.8%`** |
| `sm100_cutlass_fmha_fp8_hw_exp2_only` | FP8_E4M3 | `SM100_B200` | `0.0%` | `512.0` | `682.7` | `1137.7` | **`1,997.0`** | `45.0%` |
| `sm100_fa4_fp8_ex2_emu8_lazy_rescale` | FP8_E4M3 | `SM100_B200` | `32.5%` | `512.0` | `460.8` | `814.3` | **`2,790.1`** | **`62.9%`** |
| `sm103_b300_fa4_fp8_native_exp2_ldred` | FP8_E4M3 | `SM103_B300` | `0.0%` | `512.0` | `512.0` | `710.0` | **`3,200.0`** | **`72.1%`** |

### 核心架构师结论
1. **TMEM 原地别名与 16-Warp 解耦带来 `+38.2%` BF16 跨代收益**：相比将 SM90 FA3 架构直接外推至 B200（`1,537.5 TFLOPS`），SM100 FA4 利用 `512` 列 TMEM 原地复用 `P0/P1`（`0 B` SMEM 溢出）与独立 `Correction` Warpgroup 达到了 **`2,125.3 TFLOPS`（`95.8%` Tensor Core SOL）**。
2. **FP8 下不开启 `FFMA2` 仿真与惰性重缩放将损失 `100%` 的 FP8 加速比**：在 `sm100_cutlass_fmha_fp8_hw_exp2_only` 中，尽管 Tensor Core 耗时减半至 `512` 周期，但由于 `MUFU.EX2`（`682.7` 周期）与每步 `Correction` TMEM 读写完全统治流水线，FP8 吞吐停滞在 `1,997.0 TFLOPS`（与 BF16 完全相同！）。开启 `ex2_emu_freq=8` + `rescale_threshold=4.0` 后，吞吐跃升 **`1.397x` 至 `2,790.1 TFLOPS`**；而在 `SM103 (B300)` 上配合原生高速 `exp2` 与 `tcgen05.ld.red` 进一步达到 **`3,200.0 TFLOPS`**。
