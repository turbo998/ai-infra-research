# GPU 软件栈 Phase 169：FlashInfer Kimi-K3 896 专家融合路由、ReplaySSM Int64 扁平槽位索引、块缩放 FP8 SF 下溢防护、cuDNN Tactic 静默回退拦截及 FlashAttention SM100 `hdim=64` 反向专用 TMEM 槽位与空偏移窗口保护

## 1. 背景与核心问题

在 Blackwell (`B200`/`GB300`, `SM100`) 大规模 MoE 路由、块缩放 FP4/FP8 量化与注意力反向传播中，底层算子栈暴露出三类关键瓶颈与数值陷阱：
1. **Kimi-K3 超多专家（896 Experts, Top-16）路由开销、ReplaySSM 32 位索引溢出与块缩放 FP8 SF 下溢 `NaN`**：Kimi-K3 在每 Token 从 896 个专家中筛选 Top-16 时需要高吞吐单次启动融合路由；GDN ReplaySSM 在寻址多层检查点 `[layers * slots, HV, V, K]` 时若使用 `Int32` 步长会在大容量下发生有符号整数溢出；同时 5 处块缩放量化（Block-Scale Quantization）仅对原始 `vecMax` 判零，未对舍入后的 FP8 缩放因子（`fp8-rounded SF`）做下溢保护，当 `SF` 下溢为 `0` 时会产生 `NaN`/`Inf`。
2. **cuDNN GEMM 调优器的 `tactic=-1` 静默回退失真**：当请求的 cuDNN GEMM Tactic 不受支持时，Runner 原先静默回退执行默认 `tactic=-1`，导致 Autotuner 将回退内核耗时误记为目标 Tactic 性能。
3. **CuTe SM100 `hdim=64` 反向 TMEM 槽位争用与有意为空（Intentionally-Empty）偏移窗口被误扩为全注意力**：SM100 1-CTA `hdim=64` 反向仅使用 TMEM `[0, 384)` 列，闲置了 `[384, 512)` 空间而让 `P` 与 `dS` 复用冲突槽位；同时 `_resolve_causal_local_window` 将任何 `left + right < 0` 的非对称偏移窗口误视为 FA2 `(-1, -1)` 哨兵并重置为 `(None, None)` 全注意力。

---

## 2. 上游源码机制深挖（11 条零重叠提交）

### 2.1 FlashInfer：Kimi-K3 融合路由器、ReplaySSM `Int64` 寻址、FP8 SF 下溢防护与 cuDNN Tactic 校验
- **`e7a17d27f5` (`2026-10-29`, `#7824`)**：在 `flashinfer/experimental/kimi_k3_fused_router/` 中优化 `cake_kimi_k3_fused_router`（896 专家，Top-16），在 B200 与 GB300 上实现最高 `1.10x` 路由加速。
- **`4bddce58c3` (`2026-10-29`, `#6496`)**：在 `gdn_replayssm_spec_fold{_tcgen}.py` 中改用 `Int64` 扁平槽位索引（flat slot index）寻址全层检查点，消除 `Int32` 溢出。
- **`6622cde973` (`2026-10-29`, `#6996`)**：在 `fp4_kv_quantization.cu`、`nvfp4_smooth_quantize_sm100.cuh` 与 `fused_dit_layernorm.cuh` 等 5 处块缩放量化点直接对 `fp8-rounded SF` 做下溢判零保护，下溢时安全量化为全零而非输出 `NaN`/`Inf`。
- **`ffdf277d4a` (`2026-10-29`, `#4132`)**：在 `gemm_base.py` 与 `gemm_bf16_fp4_cudnn.py` 中当 cuDNN Tactic 不匹配或失败时显式报告 `unsupported`，彻底禁止 Profiling 阶段静默执行 `tactic=-1`。

### 2.2 FlashAttention & FlashMLA：SM100 `hdim=64` 专用 P/dS TMEM 槽位与空偏移窗口保留
- **`d86dabd8e7` (`2026-09-09`, `#2804`)**：在 `flash_attn/cute/flash_bwd_sm100.py` 中利用 1-CTA `hdim=64` 反向闲置的 TMEM `[384, 512)` 列，为 `P` 和 `dS` 分配两个专用 64 列槽位，并配合每 Warp 独立信号同步（per-warp signaling），实现 `100%`（`512/512` 列）TMEM 利用率并消除共享槽位屏障停顿。
- **`17498dad34` (`2026-09-09`, `#2490`)**：在 `flash_attn/cute/interface.py` 中修正 `_resolve_causal_local_window`，不再把 `left + right < 0` 的有意空偏移窗口扩宽为全注意力。
- **`1858932afd` & `7f55c7151a` (`2025-09-30`)**：FlashMLA 规范化错误提示信息与代码格式。

### 2.3 CUTLASS & Triton：v4.6.1 更新、循环操作数符号保留与 AMD CU-Mode 屏障测试隔离
- **`1e73948292` (`2026-07-14`, `#3383`)**：CUTLASS 发布 `v4.6.1` 更新。
- **`29ef3c3320` (`2026-10-20`, `#12350`) & `3a43613f97` (`2026-10-20`, `#12373`)**：Triton 前端在循环操作数中保留源类型有符号性（signedness），并将 AMD CU-mode 屏障测试拆分至独立文件。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/flashinfer_kimik3_router_sf_underflow_fa_sm100_tmem_phase169/run.py` 验证：

| 评估维度 | 修复/优化前基线 | Phase 169 表现 | 核心收益 |
|---|---|---|---|
| Kimi-K3 融合路由 (`896` 专家) & FP8 SF 下溢防护 | 基准路由；`5` 处 SF 下溢产 `NaN` | **`1.10x` 加速**；**`5` 处全部安全归零** | 消除块缩放量化 `NaN`、`Int32` 溢出与 `tactic=-1` 假数据 |
| CuTe SM100 `hdim=64` 1-CTA 反向 TMEM 布局 | 仅用 `384/512` 列（`75%`），P/dS 复用 | **`512/512` 列（`100%`）**，P/dS 专用 `128` 列 | 配合 Per-Warp Signaling 消除共享槽位等待 |
| CuTe 有意为空偏移窗口 (`left=-128, right=32, N=16384`) | 误扩为全注意力（计算 `16,384` tokens） | 保持空窗口（计算 **`0` tokens**） | 避免掩码稀疏注意力中的全序列无效计算与语义错误 |
