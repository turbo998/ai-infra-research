# FlashInfer FROST SM80 分组 MoE、Blackwell `cake_tinygemm2_tcgen05`、GDP 解码预取、DeepGEMM `n_l2_group=8` 聚簇对齐、KDA 分段传递、Blackwell 聚簇 Softmax 与 CUTLASS / TileLang 分析 (Phase 113 · Priority 122)

> **English Edition**: [flashinfer-frost-sm80-tcgen05-gdp-kda-cutlass-fp8-tilelang-phase113.en.md](flashinfer-frost-sm80-tcgen05-gdp-kda-cutlass-fp8-tilelang-phase113.en.md)  
> **问题 ID**: `flashinfer-frost-sm80-tcgen05-gdp-kda-cutlass-fp8-tilelang-phase113` · **优先级**: `122` · **赛道**: Track A (GPU/HBM/算子栈)  
> **确定性实验**: [`experiments/flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113/run.py`](../../experiments/flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113/run.py) (`--check` 通过)

---

## 1. 第一手源码证据矩阵（First-Hand Source Matrix）

| 序号 | 来源 ID | 仓库与提交 | 发布日期 | 证据层级 | 核心变更与代码路径 |
|---|---|---|---|---|---|
| 1 | `flashinfer-moe-frost-decode-sm80-typed-activations-2026` | [`flashinfer-ai/flashinfer@0e406f333b`](https://github.com/flashinfer-ai/flashinfer/commit/0e406f333bde1eb186023a1d28e882a235641a10) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/fused_moe/api.py`, `fma/bf16/runtime.py`：将 `FrostLowLatencyConfig` 扩展至 Ampere SM80（128-bit `cp.async` + warp MMA，`--gpu-arch sm_80` 不带 `a` 后缀）并将 SM90/SM120 分组路径推广到 FP16 与 `SwiGLU`/`GeGLU`/`ReLU2`/`SiLU` |
| 2 | `flashinfer-cake-tinygemm2-tcgen05-bf16-sm100-103-107-2026` | [`flashinfer-ai/flashinfer@8d9c839db0`](https://github.com/flashinfer-ai/flashinfer/commit/8d9c839db08d157d1bb4bd19645266813ef57d6a) | `2026-10-25` | `implementation` + `author-benchmark` | `csrc/cake_tinygemm2_tcgen05/*`：为 `sm_100a`/`sm_103a`/`sm_107a` 各生成 12 个 `tcgen05` `tinygemm_bf16` 程序（共 36 个），在 B200 `4×2880×2880` 无偏置形状上将耗时从 `8.29 us` 压至 **`7.26 us`（`1.1419x` 加速）** |
| 3 | `flashinfer-cake-gdp-preload-dense-row-t1-decode-sm100-2026` | [`flashinfer-ai/flashinfer@9f6ba5a820`](https://github.com/flashinfer-ai/flashinfer/commit/9f6ba5a820714fa30dc88bf6f6a3775819a21f40) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/cake_gdp_backend/csrc/*`：在 SM100/SM103 的单线程单行 $T=1$ GDP 解码程序中于 `k` 加载点提前发射稠密 `v`/`beta` 行预取（GB300 峰值加速 **`1.07x`**，B200 **`1.06x`**） |
| 4 | `flashinfer-cake-batch-deepgemm-fp8-even-nl2-group-k7168-2026` | [`flashinfer-ai/flashinfer@4d018bcd12`](https://github.com/flashinfer-ai/flashinfer/commit/4d018bcd12d4fc8b24f6838d0dc6c4c63a195322) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/gemm/cake_batch_deepgemm_fp8.py`：修复 `K=7168, expected_m<=1` 服务路由下 `n_l2_group=1` 导致 2-CTA 聚簇跨两个 32-row M 块混算的 bug，切换为偶数 `n_l2_group=8`（GB300 32-group 错误元素从 `97,355` 降至 **`0`**） |
| 5 | `flashinfer-cake-kda-fused-gate-segmented-16head-carry-2026` | [`flashinfer-ai/flashinfer@762b237e15`](https://github.com/flashinfer-ai/flashinfer/commit/762b237e15bdca33368889186bc26a234625e50f) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/experimental/cake_kda_train/*`：在 SM103 (`glm_intra_v14`) 将 `gate_epilogue` 融合进分块内梯度内核（反向尾端启动数 `4 -> 3`），并在 SM103/SM107 启用 16-head 分段状态递推与 SM100/SM107 双层段传递 |
| 6 | `flashinfer-cake-softmax-blackwell-cached-cluster-persistent-2026` | [`flashinfer-ai/flashinfer@510e514b25`](https://github.com/flashinfer-ai/flashinfer/commit/510e514b25f1379cc81c45cc90b82ab7d68d9b00) | `2026-10-25` | `implementation` + `author-benchmark` | `csrc/cake_blackwell_softmax_cached_cluster*.cu`：重构 7 个行缓存聚簇程序（push max/sum 交换 + 逐 tile 指数）并新增 2 个常驻聚簇路由（行数达 `2.5x` 驻留聚簇时触发；41 个形状几何平均加速 B200 **`1.283x`**、GB300 **`1.263x`**） |
| 7 | `cutlass-epilogue-per-group-norm-constant-index-fix-2026` | [`NVIDIA/cutlass@ad9bd53bda`](https://github.com/NVIDIA/cutlass/commit/ad9bd53bda6ffddbfcf6c6b7ddfd950853ee339c) | `2026-07-18` | `implementation` | `include/cutlass/epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp`, `sm120_*.hpp`：在将批坐标 `l` 归零以定位组内 SFD 指针之前保存原始 `group_idx`，修复所有分组误用 group 0 归一化常数的错误 |
| 8 | `cutlass-hopper-cutedsl-fp8-gemm-gelu-epilogue-2026` | [`NVIDIA/cutlass@a931725f1d`](https://github.com/NVIDIA/cutlass/commit/a931725f1ddc2f1dc171c22e362c4338179c6d42) | `2026-07-14` | `implementation` | `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_fp8_gelu_persistent.py`：新增参数化 Hopper CuTeDSL 常驻 FP8 GEMM + GELU 尾声融合内核 |
| 9 | `tilelang-ascend-pto-emit-ramp-typed-vectors-2026` | [`tile-ai/tilelang@d7a28bd78b`](https://github.com/tile-ai/tilelang/commit/d7a28bd78b544dcdfba2f8791808236f977f0c94) | `2026-10-19` | `implementation` | `src/ascend/codegen/codegen_pto.cc`：在昇腾 PTO 后端重写 `Ramp` 节点发射逻辑，调用 `vector_from_list` 构造带类型的单参数向量，避免元组去括号把各通道拆成多个位置参数导致 `pto.store` 编译失败 |
| 10 | `tilelang-ascend-reset-rng-state-between-kernels-2026` | [`tile-ai/tilelang@44405ee548`](https://github.com/tile-ai/tilelang/commit/44405ee54834167794bf7cb3fa334e01c2d06745) | `2026-10-16` | `implementation` | `src/ascend/codegen/codegen_ascend.cc`：在昇腾多设备内核代码生成之间重置 RNG 状态，消除跨内核随机数偏移污染 |

---

## 2. 核心实现机理与架构量化分析（Implementation Mechanics & Quantitative Analysis）

### 2.1 `cake_batch_deepgemm_fp8` 2-CTA 聚簇 M 块对齐与 `n_l2_group=8`（`4d018bcd12`）
- **故障机理**：在 `batch_deepgemm_fp8_nt_groupwise(..., backend="cake")` 的打包 `UE8M0` 服务路由（`seed_serving_m32_packed`）中，聚簇的两个 CTA 取连续 tile 索引 `block_idx` 与 `block_idx + 1`，各自在 `off_m + cta_rank * load_block_m` 加载 32 行 M 块的半区（16 行）。当 `K == 7168` 且主机提示 `expected_m <= 1` 时，`serving_config()` 误选了 `n_l2_group=1`，导致 `m_block = block_idx % m_blocks_g`。一旦某个专家实际分配到超过 32 行（例如 40 行，$\text{m\_blocks\_g} = 2$），CTA 0 计算 `m_block = 0` 而 CTA 1 计算 `m_block = 1`，把两个不同的 32 行块混在同一聚簇中累加，在 GB300（32 组）上产生 **`97,355 / 413,696` 个错误元素（最大绝对误差 `575`）**。
- **修复**：切换为偶数 `n_l2_group=8`，保证连续两个 CTA 共享同一个 `m_block = (block_idx // 8) % m_blocks_g = 0`，错误元素降至 **`0`**。

### 2.2 Blackwell 聚簇缓存 Softmax 与 CUTLASS 逐组归一化常数索引（`510e514b25` & `ad9bd53bda`）
- **Blackwell 聚簇 Softmax**：通过 push max/sum 跨 CTA 聚簇交换与逐 tile `exp2` 消除了 SM103 专用 32K 词表内核，并在 8-tile 行数达到 GPU 常驻聚簇数 **`2.5x`** 时切换到 8/16-CTA 常驻程序；41 个形状几何平均加速在 B200 达 **`1.283x`**、GB300 达 **`1.263x`**（`1 × 32000` 从 `3.26 -> 2.53 us`，**`1.291x`**）。
- **CUTLASS SM100/SM120 块缩放尾声**：在 ptr-array/grouped GEMM 的 TMA warp-specialized 访存尾声中，代码在定位每组 `SFD` 指针后执行了 `l = 0`，随后却用被清零的 `l` 索引 `norm_constant[l]`，导致所有组全部使用第 0 组的归一化常数。在清零前保存 `group_idx` 后，第 3 组缩放误差从 **`37.5`** 降至 **`0.0`**。

---

## 3. 确定性实验结果验证（Deterministic Experiment Results · `own-offline`）

运行 `python experiments/flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113/run.py --check` 的密封快照：

| 指标 | 修复前 / 基线 | 修复后 / Phase 113 | 变化率 / 结论 |
|---|---|---|---|
| FROST 编译架构标志 (`SM80` vs `SM90`/`SM120`) | 统一附加 `'a'`（`sm_80a` 非法） | **`sm_80` / `sm_90a` / `sm_120a`** | Ampere 128-bit `cp.async` + warp MMA 正常编译 |
| `cake_tinygemm2_tcgen05` `4×2880×2880` 无偏置耗时 (B200) | `8.29 us` | **`7.26 us`** | **`1.1419x` 加速**（每架构 `12` 个程序，三架构 `36` 个） |
| `cake_batch_deepgemm_fp8` `K=7168` 2-CTA M 块与坏元素 | `n_l2_group=1` (`[0, 1]`, `97,355` 坏元素) | **`n_l2_group=8` (`[0, 0]`, `0` 坏元素)** | 消除 >32 行专家聚簇错位混算 |
| `cake_blackwell_softmax` 41 形状几何平均加速 (B200 / GB300) | `1.000x` | **`1.283x` / `1.263x`** | `1×32000` 耗时 `3.26 -> 2.53 us` (`1.291x`) |
| CUTLASS 分组块缩放尾声 Group-3 缩放误差 (`ad9bd53bda`) | `37.5` (`12.5` vs `50.0`) | **`0.0` (`[12.5, 25.0, 37.5, 50.0]`)** | 捕获 `group_idx` 后再将 `l` 归零 |
| TileLang 昇腾 PTO 4 通道显式 `Ramp` 发射 (`lane=2`) | 拆成 4 个独立位置参数报错 | **`[31, 34, 37, 40]` 单向量参数** | `vector_from_list` + 跨内核 RNG 重置 |

---

## 4. 架构师设计练习与可证伪问题（Architect Exercises & Falsifiable Next Questions）

1. **计算练习**：当 `batch_deepgemm_fp8_nt_groupwise` 服务路由在 `expected_m=1` 下遇到负载倾斜（单个热专家被路由 40 个 token，即 $\lceil 40/32\rceil = 2$ 个 M 块）时，请证明为何任何奇数 `n_l2_group` 都会令 2-CTA 聚簇中的 `(2k, 2k+1)` tile 跨越不同 M 块。
2. **可证伪假设**：在 Blackwell B200 上当 `cake_blackwell_softmax` 的行数从低于 `2.5x` 常驻聚簇阈值跨越至高于 `2.5x` 阈值时，persistent 路由省去了尾波次聚簇重新调度开销，吞吐拐点可通核级 CUPTI 测量证伪。
