# FlashInfer MXFP4/MXFP8 UE8M0 次正规数量化、Rubin 2-CTA MoE 策略、Qwen 3.8 QSA 解码与 TileLang 跨作用域屏障分析 (Phase 107 · Priority 116)

> **English Version**: [flashinfer-mxfp-ue8m0-rubin-2cta-moe-qwen38-qsa-tilelang-phase107.en.md](flashinfer-mxfp-ue8m0-rubin-2cta-moe-qwen38-qsa-tilelang-phase107.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/flashinfer_mxfp_ue8m0_rubin_2cta_moe_qwen38_qsa_phase107/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`flashinfer-ai/flashinfer@fb02275` (`#7501`: `csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh`, `flashinfer/quantization/kernels/cutile/mxfp8_grouped_quantize_cutile.py`)**：
   - **MXFP4/MXFP8 统一量化与位级精确次正规数（Subnormal）处理**：新增内联 PTX `asm("mul.rn.f32 %0, %1, %2;")`（`mx_quantize_mul`）与 `ue8m0_inverse_scale(uint8_t exp)`（`exp == 0 ? 0.0f : __uint_as_float(uint32_t(max(254 - int(exp), 0)) << 23)`），即使在宿主核开启 `-use_fast_math` 时也保留 FP32 次正规数（最高至 $2^{-127}$），令零值 UE8M0 缩放字节（`e8m0_byte = 0`）在 CUDA、CuTe DSL 与 CuTile 后端间位级一致地产生带符号零载荷（`subnormal_inverse_scale = 0.0`）。
2. **`flashinfer-ai/flashinfer@03d7f4a` (`#7196`: `flashinfer/fused_moe/cute_dsl/blackwell/moe_finalize.py`, `moe_w4a16_prepare.py`)**：
   - **Split CuTe DSL W4A16 MoE 位级精确归约与 `moe_permute_w4a16`**：将 MoE Finalize 的 `_fma_rn` 从 `fma.rn.ftz.f32` 改为非 FTZ 的 `fma.rn.f32`，并抽出首项乘法 `_mul_rn` 以保留带符号零，确保 CuTe DSL W4A16 MoE 与原生 MegaMoE combine 位级对齐。
3. **`flashinfer-ai/flashinfer@b0417fd` (`#7027`: `flashinfer/fused_moe/cute_dsl/moe_utils.py`, `tuner.py`)**：
   - **Rubin (SM107) 2-CTA `tile_size=256` 与 `cluster_shape_n=2` MoE 策略**：新增 `rubin_moe_cluster_covers_one_tile(tile_size, mma_tiler, mma_inst_shape, cluster_shape_mn)`。当 `tile_size = 256`、`mma_inst_m = 256`（`atom_cta_m = 2`）且 `cluster_shape_mn = (2, 2)` 时，同一 Cluster M 维度的 2 个 CTA 映射到同一个 `moe_sort` 路由瓦片（无需额外补齐 `tile_idx_to_expert_idx` 元数据，`metadata_padding_tiles_required = 0`），同时 Cluster N 维度通过多播将激活 `A` 与缩放因子 `SFA` 的 HBM 读取带宽需求降低 **`2.0x`**。
4. **`flashinfer-ai/flashinfer@77d35fd` (`#7136`: `docs/experimental/qwen_3_8_block.md`, `flashinfer/experimental/qwen_3_5_block/_cudnn_frost/QSA/`)**：
   - **Qwen 3.8 Flash-Next 稀疏注意力（QSA）Rubin (SM107) 解码块**：基于 cuDNN Frontend `f2aa360e5`（FROST）将 QKV+Gate+Indexer 投影、4-token 块均值池化压缩键缓存写入、分页打分器与 decode top-$k$ 融合在单次解码调用中。在 `seq_len = 32,768`、`top_k_blocks = 512`（即选择 `2,048` 个稀疏 token）、`d_head = 256` 下，总 HBM 扫描流量从稠密注意力的 `33,554,432 B` 降至 **`4,194,304 B`（`8.0x` 带宽加速）**。
5. **`NVIDIA/cutlass@0efeafb` (`#3424`)、`dc45f97` (`#3556`)、`957eb92` 与 `tile-ai/tilelang@efa1ffc` (`#3643`)、`d85a70a` (`#3582`)、`d9a31dd` (`#3579`)**：
   - TileLang `ThreadPartialSyncRewriter` 在依次处理 `"shared"` 与 `"shared.dyn"` 两个存储作用域时，通过预扫描 `used_barrier_ids_` 保持部分同步命名屏障（`bar.sync id, count`）跨作用域全局唯一，将双缓冲线程组间的命名屏障冲突从 `2` 处降至 **`0` 处**（根除 warp-specialized sparse-MLA 前向核死锁）；同时修复 Ascend PTO `T.warp_reduce_max/min` 有符号归约与非零循环下界向量化。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：对于 `seq_len = 32,768`、`d_head = 256`（BF16）的 Qwen 3.8 Flash-Next 解码步，每 4 个 token 压缩为 1 个 128 维 BF16 索引键，并选取 `top_k_blocks = 512` 个块读取完整 K/V。验证为何索引键扫描字节数（`8,192 * 256 B = 2 MiB`）与稀疏 K/V 读取字节数（`2,048 * 1,024 B = 2 MiB`）之和恰好等于稠密 K/V 字节数（`32 MiB`）的 `1/8`（即 `8.0x` 有效带宽压缩）。
- **可证伪假设**：若在 Rubin (SM107) MoE 上对 `tile_size = 128`、`mma_inst_m = 128`（`atom_cta_m = 1`）强行启用 `cluster_shape_m = 2`，同一 Cluster 的两个 CTA 将跨越两个不同的路由瓦片，在 `num_non_exiting_tiles` 为奇数时导致越界读取未初始化元数据或 Cluster 屏障死锁（被 `rubin_moe_cluster_covers_one_tile == False` 证伪并拦截）。
