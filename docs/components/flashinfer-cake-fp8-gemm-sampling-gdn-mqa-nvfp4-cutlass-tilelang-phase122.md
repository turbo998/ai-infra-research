# FlashInfer CAKE Grouped FP8 GEMM 32-Row UE8M0 Swap-AB、Paged Auto Q-Len 刷新、CAKE Sampling Round 16 SM120、CAKE GDN Bwd、CAKE DeepGEMM Dense MQA、NVFP4 TMA 行边界、CUTLASS 4.8 SM107/MXFP8 与 TileLang Autotuner 深度解析（Phase 122）

> **English Version**: [flashinfer-cake-fp8-gemm-sampling-gdn-mqa-nvfp4-cutlass-tilelang-phase122.en.md](flashinfer-cake-fp8-gemm-sampling-gdn-mqa-nvfp4-cutlass-tilelang-phase122.en.md)
> **所属阶段**: Phase 122 (`priority: 131`) · Track A 算子与量化内核微架构
> **实验脚本**: [`experiments/flashinfer_cake_fp8_gemm_sampling_gdn_mqa_nvfp4_cutlass_tilelang_phase122/run.py`](../../experiments/flashinfer_cake_fp8_gemm_sampling_gdn_mqa_nvfp4_cutlass_tilelang_phase122/run.py)

---

## 1. 一手来源与证据分类（Primary Evidence）

| 证据 ID / 来源 | 仓库与 PR/Commit | 合入日期 | 证据分级 | 核心贡献 |
|---|---|---|---|---|
| `src-flashinfer-cake-grouped-fp8-gemm-ue8m0-swap-ab-small-block-sm103-pr7589-2026` | [`flashinfer-ai/flashinfer#7589`](https://github.com/flashinfer-ai/flashinfer/pull/7589) (`87658682`) | 2026-10-28 | `implementation` + `author-benchmark` | 新增 2 个 SM103 (`sm_103a`) 32-row M-block UE8M0 swap-AB 程序：`bs_ue8m0_swap_rp32_s11`（11 级流水、2-CTA cluster）与 `bs_ue8m0_swap_c1`（无 cluster 32x128 tile），在 GB300 (`152` SMs) 上将 `b3_a64_n256_k4096_p16` 提速 **1.72x** (`13.63 -> 7.93 us`) |
| `src-flashinfer-paged-auto-refresh-retained-q-len-metadata-fa2-fallback-pr7617-2026` | [`flashinfer-ai/flashinfer#7617`](https://github.com/flashinfer-ai/flashinfer/pull/7617) (`06577b40`) | 2026-10-28 | `implementation` | 修复 `BatchPrefillWithPagedKVCacheWrapper(backend="auto")` 在调用方显式传入 `block_tables` 时回退 FA2 后未刷新 `_cudnn_seq_lens_q` 的 bug，消除 SM107 上 **67.4%** 的输出失配与 NaN（`48/48` 测试格全过） |
| `src-flashinfer-cake-sampling-round16-sm120-c8-e40-radix-topk-pr7604-2026` | [`flashinfer-ai/flashinfer#7604`](https://github.com/flashinfer-ai/flashinfer/pull/7604) (`9657c04e`) | 2026-10-27 | `implementation` + `author-benchmark` | `cake_sampling` Round 16 为 RTX PRO 6000 (`sm_120a`, `188` SMs, `101,376 B` 动态 SMEM) 新增 `(8, 40)` stage-1 radix top-k 内核（`106 -> 108` builds），在 `V=151,936` 实现 **92.7%** 寄存器驻留且零 spill，最高提速 **1.151x** |
| `src-flashinfer-cake-gdn-train-bwd-swizzled-dqkg-tape-prefetch-sm107-pr7602-2026` | [`flashinfer-ai/flashinfer#7604` / `#7602`](https://github.com/flashinfer-ai/flashinfer/pull/7602) (`9c81aefd`) | 2026-10-27 | `implementation` + `author-benchmark` | `cake_gdn_train` 反向 `dqkg` XOR lane swizzle + 128B 打包写回、`kcp_dhu_tape` 提前 1 步 `2 x 256-bit` 预取 `dv_intra`，并将 SM107 `bwd_prep` 设为 `ctas_per_sm=2` (`148 -> 296` CTAs)，GB300 P1 提速至 `24.733 ms` (**1.022x**) |
| `src-flashinfer-cake-deepgemm-dense-mqa-warp-reduction-sm100-sm103-pr7607-2026` | [`flashinfer-ai/flashinfer#7607`](https://github.com/flashinfer-ai/flashinfer/pull/7607) (`5073cd43`) | 2026-10-27 | `implementation` + `author-benchmark` | `cake_deepgemm_dense_mqa` 在 `sm_100a`/`sm_103a` 替换 15 个稠密 MQA logits 内核（单轮 warp 扫描 `768` 块、融合 `f32x2` abs+add/FMA、移除冗余 `tcgen05` 栅栏），B200 中位提速 **1.0537x**（最高 **1.1199x**），GB300 中位 **1.0458x**（最高 **1.1012x**） |
| `src-flashinfer-nvfp4-quantize-tma-scale-factor-row-bound-pr6278-2026` | [`flashinfer-ai/flashinfer#6278`](https://github.com/flashinfer-ai/flashinfer/pull/6278) (`e9c4787c`) | 2026-10-27 | `implementation` | 在 `quantize_with_block_size_tma` 中增加 `threadRowIdxGlobal < numPaddedRowsForSf` 守卫，消除 `m % 16 in [1, 8]` 时 `layout_8x4` 越界写入 `n / 2` 字节（`n=2048` 时 `1024 B`）并移除 Python 侧 `round_up(m, 16)` 冗余分配 |
| `src-cutlass-v48-rubin-sm107-mxfp8-k64-sf-k128-tile-pr3645-2026` | [`NVIDIA/cutlass#3645`](https://github.com/NVIDIA/cutlass/pull/3645) (`098de2a6`) | 2026-09-22 | `implementation` | CUTLASS 4.8.0 引入 Rubin (`SM107`) CuTe DSL 扩展（`MMA_K=64` FP8、`MMA_K=128` FP4、`576` TMEM 列、`328 KB` SMEM）与 `is_mxfp8_k64_sf_k128_tile`（两个 `K=64` 迭代共享一次 `K=128` SF 加载） |
| `src-tilelang-autotuner-get-tunner-forward-profile-args-pr3337-2026` | [`tile-ai/tilelang#3337`](https://github.com/tile-ai/tilelang/pull/3337) (`90f8908b`) | 2026-10-19 | `implementation` | 修复 `AutoTuneImpl.get_tunner()` 未向 `set_profile_args()` 转发 `warmup, rep, timeout` 导致自定义调优预算全部碰撞到默认 `(25, 100, 30)` 缓存键 (`hash=74468074`) 的问题 |

---

## 2. 核心架构机制与公式推导

### 2.1 CAKE Grouped FP8 GEMM 32-Row M-Block UE8M0 Swap-AB (`#7589`)

在 Blackwell Ultra (`sm_103a`, GB300 `152` SMs) 上，当分组 FP8 GEMM 采用 1D1D UE8M0 块缩放且 token 数 `M` 较小、输出通道 `N` 较窄（如 `N=256..1024`）时，原有的 64-row 或 128-row tile 会导致严重的专家尾部填充浪费与 CTA 数量不足。PR `#7589` 新增两个 32-row swap-AB（在张量核 MMA 中将激活 `A` 作为 `B` 矩阵、权重 `B` 作为 `A` 矩阵以获得 32-row M 步进）程序：

1. **`bs_ue8m0_swap_rp32_s11`（11 级流水，2-CTA Cluster）**：
   $$\text{CTAs}_{\text{rp32\_s11}}(M, N) = 2 \cdot \left\lceil \frac{M}{32} \right\rceil \cdot \frac{N}{256} \le \text{SM\_count}, \quad K \ge 2048$$
   在 `M=2432, N=256, K=4096` 边界点，$\text{CTAs} = 2 \cdot 76 \cdot 1 = 152$，恰好在 GB300 (`152` SMs) 上单波（1-wave）满载运行，而在 `148` SMs 设备上自动回退。在 `b3_a64_n256_k4096_p16` (`M=2048, routed=1417, N=256, K=4096`) 上，耗时从 `13.63 us` 降至 `7.93 us`（**1.72x**）。
2. **`bs_ue8m0_swap_c1`（单 CTA 32x128 tile，无 Cluster）**：
   $$\text{CTAs}_{\text{c1}}(M, N) = \left\lceil \frac{M}{32} \right\rceil \cdot \frac{N}{128} \le \text{SM\_count}, \quad \text{align}=32, \; 256 \le K \le 512, \; N \le 512$$

### 2.2 Paged `backend="auto"` Q-Length 元数据刷新 (`#7617`) 与 NVFP4 TMA 行边界 (`#6278`)

- **Paged Auto 保留元数据刷新 (`#7617`)**：当调用方在首次 `plan()` 显式传入 `block_tables` 时，`self._cudnn_block_tables` 保持 `None`（复用外部表），但 `self._cudnn_seq_lens_q` 会被分配并记录首次 `qo_indptr`。若第二次 `plan()` 因变长 Q 回退至 FA2，原有守卫 `if self._cudnn_block_tables is not None:` 跳过了 `_cudnn_seq_lens_q` 更新；当第三次 `plan()` 重新回到 cuDNN 时，cuDNN 读取陈旧的首次 Q 长度，在 SM107 上造成 **67.4%** 的输出错误与 NaN。改为检查 `if self._cudnn_seq_lens_q is not None:` 后，`48/48` 测试矩阵全部通过。
- **NVFP4 TMA 缩放因子越界保护 (`#6278`)**：`quantize_with_block_size_tma` 对 FP16/BF16 启动 `TMA_ROW_TILE = 16` 行的网格，但 `SFMatrixLayout::layout_8x4` 仅将缩放因子分配对齐到 `8` 行（`numPaddedRowsForSf = ceil(m / 8) * 8`）。当 `m % 16 in [1, 8]`（如 `m = 1025`）时，启动行数为 `1040` 而 SF 仅分配 `1032` 行，多出的 `8` 行会向 SF 缓冲区末尾之后越界写入 $8 \times (n / 16) = n / 2$ 字节（`n=2048` 时越界 `1024 B`）。新增 `threadRowIdxGlobal < numPaddedRowsForSf` 彻底消除了越界写入。

### 2.3 CAKE Sampling Round 16 SM120 (`#7604`)、CAKE GDN Bwd (`#7602`)、Dense MQA (`#7607`)、CUTLASS 4.8 (`#3645`) 与 TileLang (`#3337`)

- **CAKE Sampling Round 16 (`#7604`)**：在 RTX PRO 6000 (`sm_120a`, `188` SMs, `101,376 B` 动态 SMEM) 上，针对 Qwen 词表 `V = 151,936`（位于 `(131072, 163840]` 区间），新增 `(cluster=8, ept=40, threads=512)` 内核，最大词表容量 $8 \times 512 \times 40 = 163,840$，寄存器有效驻留率从 `(8, 48)` 的 `77.3%` 提升至 **92.7%** 且消除局部内存 spill，在 `top_k=10, batch=4` 提速 **1.151x**；同时对 `batch >= 160` (`ept=40`) 切换至 2-launch 链以避免 `cluster=8` 槽位饥饿。
- **CUTLASS 4.8 SM107 & MXFP8 (`#3645`) 与 TileLang Autotuner (`#3337`)**：CUTLASS 4.8 引入 Rubin `SM107`（`576` 列 TMEM、`328 KB` SMEM、`MMA_K=64` FP8 / `128` FP4）及 `(128, 256, 64)` MXFP8 tile 的双迭代共享 `K=128` SF 块；TileLang `AutoTuneImpl.get_tunner()` 补齐 `set_profile_args(warmup, rep, timeout)` 转发，防止不同调优预算共享同一个缓存哈希。
