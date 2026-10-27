# FlashInfer cuDNN 计划复用、Cake Ulysses 直写 NVLink、GEMM 宿主分派消减、MiniMax-H3 SM120 与 CUTLASS/TileLang 分析 (Phase 109 · Priority 118)

> **English Version**: [flashinfer-cudnn-plan-ulysses-gemm-host-minimax-cutlass-tilelang-phase109.en.md](flashinfer-cudnn-plan-ulysses-gemm-host-minimax-cutlass-tilelang-phase109.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/flashinfer_cudnn_plan_ulysses_gemm_host_minimax_phase109/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`flashinfer-ai/flashinfer@5a06916e7b` (`#7052`: `flashinfer/prefill.py`)**：
   - **cuDNN 分页 `plan()` 元数据复用与冗余上传跳过**：常规 cuDNN 分页 `plan()` 此前每次调用都会上传两条执行路径从未读取的 FA2 元数据数组（`page_indptr` 与 `last_page_len`），且显式 cuDNN 在每次 `plan()` 时都会重新分配 CPU 源 `qo_indptr` 前缀缓冲并重建静态绑定。修复后将条件统一为 `if self._backend == "cudnn" and qo_indptr.device.type == "cpu":`，复用包装器持有的 Q 前缀缓冲并在非 CUDA Graph 模式下跳过两条冗余上传，将每次 `plan()` 的 H2D 元数据上传从 `3` 次降至 **`1` 次（削减 `66.67%`）**，128 次规划中的 Q 前缀重分配从 `128` 次降至 **`1` 次**。
2. **`flashinfer-ai/flashinfer@35591b5d6a` (`#7513`: `flashinfer/cake_ulysses_backend/`)**：
   - **Cake Ulysses 序列并行 NVLink 直写 All-to-All 内核优化**：`flashinfer.comm` 的 Ulysses 序列并行（跨 NVLink 散射/聚集注意力头）共包含 `48` 个程序（`bf16/f16/f32` × `world_size 2/4/6/8` × `2` 方向 × `staged/direct` 双路径）。本次更新全部 **`24` 个 direct-write 程序**，使每个 CTA 直接通过 NVLink 写入对端由 `allocate_output` 注册的目标显存缓冲区，消除分阶段中转缓冲区的二次全局显存写入（`2.0x` 全局写入流量削减，单跳延迟从 `148.6 us` 降至 **`125.9 us`，`1.1803x` 加速**）。
3. **`flashinfer-ai/flashinfer@946eabe5a5` (`#7485`) 与 `ecc3b2f8e3` (`#7492`)**：
   - **Cake Top-k/Top-p 采样第 14 轮优化与 MXFP8 门控激活带状 Sigmoid**：`#7485` 将 `cake_sampling` 的 Stage-1 基数选择内核构建数从 `104` 扩展至 **`106` 个**（配合 `15` 个 Stage-2/3 稀疏重归一化采样构建），针对 `top_k = 1000` 适配候选列表并在 `vocab_size = 262,144, batch_size = 128` 下实现单次启动（`2 -> 1` 次启动）全 CTA 尾部归约；`#7492` 为 `silu_and_mul_mxfp8_quantize`（输入 `[M, 2K]` BF16 -> 输出 `[M, K]` E4M3 + 每 32 元素 1 个按 `128x4` 交错排布的 E8M0 缩放因子）在 `SM100/SM103/SM107` 上引入带状路径（Band-Path）受保护 Sigmoid，将前/反向融合量化延迟从 `41.2 us` 降至 **`36.8 us`（`1.1196x` 加速）**。
4. **`flashinfer-ai/flashinfer@4dd1d9b964` (`#4674`: `flashinfer/gemm/gemm_base.py`) 与 `4d1799541e` (`#7487`: `csrc/cake_minimax_h3_sm120_nvfp4_varlen_attention/`)**：
   - **GEMM 宿主端每次调用分派开销从 `200 us` 降至 `59 us` + SM120 MiniMax-H3 NVFP4 变长注意力重叠预处理**：`#4674` 严格划清规划期与调用期边界，将仅依赖 `(dtype, layout, n, k, m_bucket, device)` 的解析缓存化，使小批量在线推理（`M = 8, N = K = 2048`，GPU 内核仅约 `2.0 us`）下 `mm_bf16` 的 CPU 分派开销从 `200.0 us` 降至 **`59.0 us`（`3.3898x` 提速，削减 `70.5%` 宿主端开销）**；`#7487` 为 `sm_120a` 交付带均值补偿与无增量两套 SageAttention3 NVFP4-QK 变长注意力（`[tokens, 56 heads, 128 head_dim]`），重叠预处理后延迟从 `94.0 us` 降至 **`81.5 us`（`1.1534x` 加速）**。
5. **`NVIDIA/cutlass@564d267e4c` (`#3461`)、`b50d8fd719` (`#3394`) 与 `tile-ai/tilelang@c362c962d0` (`#2815`)、`0bd33f953c` (`#3511`)**：
   - CUTLASS `#3461` 修复 `sm100a` 块缩放分布式 GEMM All-Reduce（`ldmcxstmc`）内核签名，移除冗余 `lambda x: x` 位置参数；`#3394` 在 `GemmUniversalBlockwise` 中向 `mma(...)` 显式传入 `threadblock_tile_offset.m()` 与 `threadblock_tile_offset.n()`，修复 SM89 FP8 块级交错缩放因子索引在多 Tile 网格（`4x4 = 16` 个 Tile）下全部混叠至 `(0, 0)` 的算术错误。TileLang `#2815` 支持 `T.reduce` 的 `keepdim=True` 同秩目标寄存器片段布局归一化（`(M, 1)` vs `(M,)`）；`#3511` 为昇腾（Ascend）后端支持带显式 `l0_pitch` 的部分 L0 拷贝及精确 L1 写入依赖分析。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：当在线解码小批量 GEMM 形状为 `M = 8, N = K = 2048`（GPU 执行耗时 $T_{\text{gpu}} = 2.0\text{ }\mu\text{s}$）时，对比优化前宿主分派耗时 $T_{\text{host,old}} = 200.0\text{ }\mu\text{s}$ 与缓存 `(dtype, layout, n, k, m_bucket, device)` 后的 $T_{\text{host,new}} = 59.0\text{ }\mu\text{s}$，计算无 CUDA Graph 时 GPU 有效繁忙占比的变化（从 $2/200 = 1.0\%$ 提升至 $2/59 = 3.39\%$，吞吐上限提升 $3.3898\times$）。
- **可证伪假设**：若在 SM89 FP8 块级 GEMM 的 `GemmUniversalBlockwise::mma` 调用中省略 `threadblock_tile_offset.m(), threadblock_tile_offset.n()`，则所有线程块都将读取左上角首个 Tile `(0, 0)` 的 `scale_A / scale_B` 因子，导致非零偏移 Tile 出现严重反量化数值失真（已被 `cutlass #3394` 单测验证并修复）。
