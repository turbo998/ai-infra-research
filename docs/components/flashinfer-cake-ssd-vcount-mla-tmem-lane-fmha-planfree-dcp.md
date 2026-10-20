# FlashInfer Cake SSD Look-Back 虚拟 CTA 上限、Blackwell MLA TMEM Layout-B 1/4 通道寻址与 FMHA 免规划静态单波次 DCP 解码深度剖析

> **Lang / 语言**: **中文 (当前)** | [English](./flashinfer-cake-ssd-vcount-mla-tmem-lane-fmha-planfree-dcp.en.md)
>
> **证据等级 (Evidence Class)**：`implementation`（上游固定 Commit 源码核验）+ `author-benchmark`（上游作者在 NVIDIA Rubin R200 208-SM、GB300 SM103a 与 B200 SM100a 上的实测数据，本项目未重测 `independently_reproduced=false`）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/flashinfer_cake_ssd_vcount_mla_tmem_fmha_planfree/run.py --check`）。

---

## 1. 架构背景与核心问题

在 NVIDIA Blackwell（`sm_100a` B200 148 SMs、`sm_103a` GB300 152 SMs）与 Rubin（`sm_107a` R200 208/212/216 SMs）代际的持久线程块（Persistent 1-CTA-per-SM）算子工程中，主机侧加载器（Loader）、张量内存（TMEM `tcgen05`）寻址与设备端任务规划器（Device Planner）之间的契约决定了延迟下界与数值正确性：

1. **Mamba2 SSD Look-Back 跨板卡变体的二波次长尾（Tail-Wave Straggler）**：FlashInfer `SSDCombined(backend="cake")` 的 Look-Back 块并行预填充程序（`chunkpar_*`）将 tile 序列切分为 $V$ 个虚拟 CTA 区间（Virtual CTA Ranges），并由 `1-CTA-per-SM` 持久网格通过原子票据抢占直到耗尽。旧版主机加载器固定传入架构表值 $V_{\text{table}}$（`sm_100a=148`、`sm_103a=152`、`sm_107a=212`）。当运行在 **208 SM 的 Rubin R200 板卡变体**上时，208 个物理 CTA 领完首轮 208 个区间后，剩余 $212 - 208 = 4$ 个区间迫使 4 个 CTA 执行**第二轮完整计算与前缀归约**，其余 204 个 SM 空转等待，导致整算子耗时膨胀 $1.35\times\sim 1.64\times$。
2. **Blackwell `cta_group::2` MLA 解码 TMEM Layout-B 分数跨槽错读**：在 `trtllm_batch_decode_with_kv_cache_mla(backend="cake")` 的 `mla_bf16_native_split8_pdl` 路由及 `cake_mla_varq_dcp_decode` 的 8 个内核中，分数 MMA 采用 `M128 N128 cta_group::2` 并维护两个相位槽（TMEM 列 `0` 与 `64`）。在 Layout-B 下，行 $r$ 的 `keys 64..127` 存放在 TMEM 通道 `64 + r` 且与 `keys 0..63` **共享相同列偏移**，每个 softmax warp 读取各自的 32 通道四分之一区（`((warp % 4) * 32) << 16`）。旧版代码误加了列偏移 `n_half * 64`，导致处理 `keys 64..127`（`n_half = 1`）的 softmax warps 2 与 3 跨槽读取了相邻相位槽的陈旧分数。更隐蔽的是，单元测试原先使用 `0.05` / `0.1` 的小幅随机输入，使 softmax 近似均匀分布，错读误差被掩盖在 `atol=1e-2` 之内。
3. **E4M3 `head_dim=128` 均衡 DCP 投机解码的设备规划器开销**：`cake_fmha` 的 `dcp_spec_bf16_fp8_balanced` 家族此前仅包含 Program 0（`at_or_above_grid`，无 8-slice 折叠）和 Program 1（`below_grid`，含设备端动态规划器与空闲 CTA 8-slice 折叠）。对于单批次中等上下文投机解码（如 `prod_b1_s8192_q4_cp4_graph`），所有块票据与归约票据完全可容纳在单波次网格内，设备端规划器的 `mbarrier` 同步与动态票据派生属于纯开销。

---

## 2. 一手源码实现路径剖析

### 2.1 Cake SSD Combined Look-Back 虚拟 CTA 上限：`min(table_vcount, sm_count)`

在 `flashinfer-ai/flashinfer` Commit [`4af20723d2`](https://github.com/flashinfer-ai/flashinfer/commit/4af20723d27e650b67dbb20b0569c316b2185ded)（PR `#6884`，2026-10-20）中，`flashinfer/mamba/cake_ssd_combined.py` 将 `_chunk_parallel_vcount` 升级为感知设备实际 `sm_count`：

```python
_CHUNK_PARALLEL_VCOUNT_BY_ARCH = {
    (10, 0): 148,
    (10, 3): 152,
    (10, 7): 212,
}

def _chunk_parallel_vcount(
    capability: tuple[int, int], sm_count: Optional[int] = None
) -> int:
    key = (int(capability[0]), int(capability[1]))
    vcount = _CHUNK_PARALLEL_VCOUNT_BY_ARCH.get(
        key, _CHUNK_PARALLEL_VCOUNT_BY_ARCH[_CHUNK_PARALLEL_VCOUNT_DEFAULT]
    )
    if sm_count is not None:
        vcount = min(int(vcount), int(sm_count))
    return max(1, min(int(vcount), _CHUNK_PARALLEL_SEGMENT_CAP))
```

- **确定性契约与 CUDA Graph 兼容性**：每个浮点结果仅取决于 `(inputs, V)` 而与启动网格无关；`agg_work` 工作区仍按架构表上限 `212 * (64 * 128 + 1)` 分配，内核仅索引前 $V$ 个区间。由于单设备的 `sm_count` 恒定，CUDA Graph 捕获后重放的 $V$ 完全一致。
- **跨板卡行为**：208-SM Rubin R200 绑定 $V = \min(212, 208) = 208$（每个 CTA 严格 1 个区间）；212/216-SM R200 保持 $V = 212$、148-SM B200 保持 $V = 148$、152-SM GB300 保持 $V = 152$，字节级完全不变。

### 2.2 Blackwell MLA TMEM Layout-B 1/4 通道寻址修复

在 `flashinfer-ai/flashinfer` Commit [`bc45bb9878`](https://github.com/flashinfer-ai/flashinfer/commit/bc45bb9878d6153fc83baa38c2ca1a5e9b3db446)（PR `#6590`，2026-10-20）中，修复了 `csrc/cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_db015a53334184a3005e_kernel.cu` 与 `flashinfer/experimental/cake_mla_varq_dcp_decode/csrc/cake_mla_varq_dcp_decode/*_kernel.cu` 共 **9 个内核源文件**中的 `tcgen05.ld.sync.aligned.32x32b.x32.b32` 基址计算：

```cpp
// 修复前（错误叠加 n_half * 64 列偏移，导致 warps 2 & 3 读入另一相位槽）：
int s_base = taddr + (unsigned int)s_off + (unsigned int)(n_half * 64) + (unsigned int)(tmem_row_base << 16);

// 修复后（直接按 warp 所属 1/4 通道偏移定址当前相位槽列 s_off）：
int s_base = taddr + (unsigned int)s_off + (unsigned int)(((warp % 4) * 32) << 16);
```

- **TMEM 地址编码原理**：Blackwell TMEM 地址低 16 位表示列偏移（Column Offset，双缓冲槽分别为 `0` 和 `64`），高 16 位（`<< 16`）表示通道（Lane `0..127`）。Layout-B 将 `keys 0..63` 置于通道 `0..63`（由 `warp % 4 in {0, 1}` 读取），将 `keys 64..127` 置于通道 `64..127`（由 `warp % 4 in {2, 3}` 读取），二者位于**同一列槽 `s_off`**。
- **测试输入幅度下界**：当输入缩放因子为 `0.05` 或 `0.1` 时，logits 方差极小，$\text{softmax}(\text{logits}/\sqrt{512})$ 接近 $1/K$ 均匀分布，跨槽错读的加权和偏差小于 `1e-2`；将测试输入缩放提升至 `_INPUT_SCALE = 0.5` 后，峰值注意力分布立即暴露旧版在 `kv_lens=(1024,)` Split-8（每 split 128 keys > 64）下高达 `3.2e-2` 的最大绝对误差（`12.2%` 元素超差），修复后最大绝对误差降至 `3.1e-4`，且 GB300 延迟保持 `11.84 µs`（零性能回退）。

### 2.3 `cake_fmha` E4M3 `head_dim=128` 免规划静态单波次（Program 2）

在 `flashinfer-ai/flashinfer` Commit [`fda1694904`](https://github.com/flashinfer-ai/flashinfer/commit/fda16949043745adcf03ab4b598c977c5b91c3b6)（PR `#6119`，2026-10-20）中，`flashinfer/cake_dcp.py` 的 `dcp_balanced_program` 为 `dcp_spec_bf16_fp8_balanced` 引入三分支选择规则：

$$
n_{\max} = \left\lceil \frac{\text{max\_pages\_per\_seq} \times 64}{256} \right\rceil, \quad T_{\text{total}} = B \cdot H_{kv} \cdot (n_{\max} + 8)
$$

1. **Program 0 (`at_or_above_grid`)**：当 $B \cdot H_{kv} \ge \text{sm\_count}$ 时启用，每个 `(request, KV head)` 至少占 1 张块票据，无空闲 CTA 可执行 8-slice 折叠，直接运行无折叠体的设备规划器。
2. **Program 2 (`static_one_wave`)**：当 $B \cdot H_{kv} < \text{sm\_count}$、且 $n_{\max} \ge 3$（至少 3 个 256-token 块）、且总票据数 $T_{\text{total}} \le \text{sm\_count}$ 时启用。此时每个 CTA 直接以 `blockIdx` 作为静态票据（`ticket = CTA index`），**完全跳过设备端规划器（Plan-Free）**，输出与 Program 1 逐位一致（Bitwise Identical）。
3. **Program 1 (`below_grid`)**：其余场景默认运行带 8-slice 折叠的设备规划器。

---

## 3. 定量基准与确定性实验验证

### 3.1 作者实测数据（`author-benchmark`，`independently_reproduced=false`）

1. **Rubin R200 (208 SMs, `sm_107a`) Mamba2 Cake SSD Look-Back (`4af20723d2`)**：
   - `1x8192, h=8, bf16`（512 tiles，$V: 212 \to 208$）：Warm `61.39 -> 40.85 µs`（**1.503x**），Cold `71.09 -> 44.93 µs`（**1.582x**）。
   - `1x32768, h=8, bf16`（2048 tiles，$V: 212 \to 208$）：Warm `164.55 -> 104.94 µs`（**1.568x**），Cold `167.02 -> 107.25 µs`（**1.557x**）。
   - `1x65536, h=8, bf16`（4096 tiles，$V: 212 \to 208$）：Warm `301.68 -> 183.87 µs`（**1.641x**），Cold `303.55 -> 185.15 µs`（**1.639x**）。
   - `2x256, h=128, bf16`（512 tiles，$V: 212 \to 208$）：Warm `47.73 -> 35.50 µs`（**1.345x**），Cold `59.20 -> 39.15 µs`（**1.512x**）。
2. **B200 (`sm_100a`, 148 SMs) & GB300 (`sm_103a`, 152 SMs) `cake_fmha` Program 2 (`fda1694904`)**：
   - `prod_b1_s8192_q4_cp4_graph`（`max_pages_per_seq=36`，$n_{\max}=9$，$T_{\text{total}}=8 \times (9+8)=136 \le 148$）：B200 延迟从 `12.58 µs` 降至 **`10.85 µs`（1.159x）**；GB300 延迟从 `12.58 µs` 降至 **`11.87 µs`（1.059x）**。

### 3.2 本项目离线确定性实验（`own-offline`）

运行 `python experiments/flashinfer_cake_ssd_vcount_mla_tmem_fmha_planfree/run.py --check` 验证了：
- 8 种架构/SM 组合下 `_chunk_parallel_vcount` 的精确映射与二波次 CTA 消除（R200-208SM `second_round_ctas: 4 -> 0`）；
- Layout-B TMEM 4 个 softmax warps 在双相位槽下的精确地址生成与 `input_scale=0.05/0.1/0.5` 的注意力误差放大效应（`0.05` 误差 `0.002016 < 1e-2` 被掩盖 vs `0.50` 误差 `0.483254 > 1e-2` 显著暴露）；
- 8 个生产契约形状在 B200（148 SMs）与 GB300（152 SMs）上的 `dcp_balanced_program` 路由一致性。

---

## 4. 面向解决方案架构师的设计准则与可证伪问题

1. **持久内核跨 SKU 迁移必须校验“虚拟分片数 $V \le \text{物理 SM 数}$”**：在同一计算能力（如 `sm_107a` Rubin 或 `sm_100a` Blackwell）下，云厂商常交付不同 SM 数的缩减版（Sub-binned SKU，如 208 SM vs 212/216 SM）。任何硬编码架构表值 $V_{\text{table}}$ 的持久网格算子都会在缩减版 SKU 上触发 $V_{\text{table}} - \text{SM}$ 个 CTA 的尾部第二波次，造成高达 35%~64% 的性能劣化。
2. **算子数值回归测试严禁使用过小输入方差**：注意力内核（尤其是 FP8/BF16 多槽流水线）的单元测试若使用 `randn * 0.05`，会使 softmax 退化为均匀平均器，掩盖高达 50% 通道的跨槽错读。测试输入幅度应至少令 $\max(\text{logits}) - \min(\text{logits}) \ge 2.0$。
3. **后续可证伪问题**：当 `cake_fmha` 扩展至 `head_dim=256`（`dcp_spec_bf16_fp8_d256_balanced`）时，单波次静态分配器（Program 2）的寄存器压力是否仍允许保持相同的 `reduce_tickets_per_tile = 8` 阈值？
