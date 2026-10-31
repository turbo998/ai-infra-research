# GPU 软件栈 Phase 168：FlashInfer Frost FP4 宽加载解码与无效路由掩码、Rubin SM107 MegaMoE 融合量化、Cake TinyGEMM2 `PACKED_M64_N8` 及 FlashAttention SM90 反向与因果滑动窗口修复

## 1. 背景与核心问题

在 Blackwell (`SM100a`/`SM103a`)、Rubin (`SM107`) 与 Hopper (`SM90`) 算子演进中，低延迟 FP4 MoE 与注意力内核暴露出三类隐蔽缺陷：
1. **Frost 低延迟 MoE 的 8 值宽加载错位与无效路由 `NaN` 污染**：在 SM100 上使用原生 8 值 NVFP4/MXFP4 权重宽加载时，打包半字（packed halfwords）解码错位会导致输出失真；同时 MXFP8/MXFP4 归约阶段将零部分和与无效路由的 `NaN` 权重相乘（`0.0 * NaN = NaN`），导致整条 Token 输出被污染。
2. **Rubin `SM107` MegaMoE 多核启动开销与 `tcgen05` 小批量 GEMM Tile 缺失**：MegaMoE 解码路径原先将路由合法性校验、MXFP8 输入量化与 `TopkReduce` 暂存拆分为 3 次独立启动；同时 `cake_tinygemm2_tcgen05` 需要在 `sm_100a`/`sm_103a` 上补齐 `PACKED_M64_N8` 内核与无偏路由窗口。
3. **CuTe SM90 `head_dim ∈ (128, 192]` 反向崩溃与因果滑动窗口 `window_size_right=0` 误判**：当启用 `dKV_swapAB=True` 时，`head_dim` 映射到 SM90 BF16 WGMMA 的 `M=64` 轴，旧版按 16 对齐（如 `hdim=160`）会导致 CuTe Tiled-MMA 划分失败崩溃；此外 Python 表达式 `(window_size_right or max_seqlen_k)` 将因果滑动窗口的 `window_size_right=0` 误判为布尔假（falsy），导致调度器按全序列 `max_seqlen_k` 过度规划 KV 加载。

---

## 2. 上游源码机制深挖（11 条零重叠提交）

### 2.1 FlashInfer：Frost FP4 修复、Rubin SM107 MegaMoE 融合与 Cake TinyGEMM2 `tcgen05`
- **`bd460fcb23` (`2026-10-29`, `#7845`)**：在 `flashinfer/fused_moe/backends/frost_low_latency/fma/fp4_weight.py` 中通过显式字节提取（explicit byte extraction）解码两个打包半字，修复 8 值宽加载下的 NVFP4/MXFP4 权重解码错误。
- **`f416759b9c` (`2026-10-29`, `#7838`)**：在 `fma_common.py` 与 `fma_fc2.py` 中对无效 Frost 路由显式掩码（避免 `0 * NaN` 污染）并修复 MXFP4 FFMA 解码。
- **`8cc2453f02` (`2026-10-29`, `#7806`)**：在 `flashinfer/moe_ep/backends/mega/kernel/sm107/mxfp8_mxfp8_bf16_cutedsl/` 中将路由校验融合进 MXFP8 输入量化，并令 `TopkReduce` 直接写入调用方对齐的连续输出缓冲区，将启动次数从 `3` 降至 `1`。
- **`9318fc8621` (`2026-10-29`, `#7826`)**：在 `csrc/cake_tinygemm2_tcgen05/` 下为 `sm_100a`/`sm_103a` 构建全部 `36` 个 `PACKED_M64_N8` 变体并加入无偏路由窗口。

### 2.2 FlashAttention & FlashMLA：SM90 反向 `M=64` 对齐与滑动窗口零边界修复
- **`9d61d35ba8` (`2026-09-09`, `#2482`)**：在 `flash_attn/cute/flash_bwd_sm90.py` 中将 `dKV_swapAB=True` 且 `head_dim ∈ (128, 192]` 时的 `tile_hdim` 向上对齐至 `64` 的整数倍（`192`），满足 SM90 WGMMA `M=64` 约束。
- **`d8dbfad57d` (`2026-09-09`, `#2624`)**：在 `flash_attn/cute/interface.py` 中显式判断 `window_size_right < 0` 而非使用 `or`，修复 `window_size_right=0` 被当作无界窗口的问题。
- **`082094b793` (`2026-01-16`, `#150`) & `1408756a88` (`2025-09-30`)**：FlashMLA 完成多处内核重构与架构文档更新。

### 2.3 CUTLASS & Triton：AR RS 更新、返回元组长度校验与 AMD 单内核 `clopt`
- **`a3a1b5cfe6` (`2026-07-16`, `#3381`)**：CUTLASS 更新 AllReduce + ReduceScatter（AR RS）融合示例。
- **`7302258a3e` (`2026-10-20`, `#12381`) & `9b12eff0a6` (`2026-10-20`, `#12382`)**：Triton 前端拒绝函数不同分支返回不等长元组，并为 AMD 后端新增每内核编译器选项（per-kernel `clopt`）控制。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/flashinfer_frost_fp4_rubin_megamoe_fa_sm90_bwd_phase168/run.py` 验证：

| 评估维度 | 修复/优化前基线 | Phase 168 表现 | 核心收益 |
|---|---|---|---|
| Frost FP4 宽加载与无效路由 (`8` lanes) & Rubin MegaMoE | 半字错位 + `NaN` 污染；`3` 次启动 | **`0` 错位 / `0` `NaN`**；**`1` 次融合启动** | 启动开销压缩 **`3.0x`**；`36` 个 `tcgen05` 变体就绪 |
| CuTe SM90 反向 (`head_dim=160, dKV_swapAB=True`) | `tile_hdim=160` (`160 % 64 = 32` 崩溃) | `tile_hdim=192` (**`192 % 64 = 0`**) | 彻底消除 `(128, 192]` 维度反向 WGMMA 崩溃 |
| CuTe 因果滑动窗口 (`max_k=128K, left=4096, right=0`) | 误规划加载 `135,168` tokens | 精确规划加载 **`4,096` tokens** | **KV 调度过度规划消除 `33.0x`** |
