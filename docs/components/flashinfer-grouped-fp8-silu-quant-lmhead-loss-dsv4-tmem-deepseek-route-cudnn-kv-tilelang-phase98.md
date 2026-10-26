# FlashInfer Grouped FP8 Fused SiLU-Quant Pair+Tail 与原位绑定、LM-Head Loss 单同步压缩、DSv4 TMEM 1/4 通道隔离、DeepSeek 路由边界、cuDNN 松弛 KV 上界与 TileLang 演进（Phase 98）

> **English Version**: [flashinfer-grouped-fp8-silu-quant-lmhead-loss-dsv4-tmem-deepseek-route-cudnn-kv-tilelang-phase98.en.md](flashinfer-grouped-fp8-silu-quant-lmhead-loss-dsv4-tmem-deepseek-route-cudnn-kv-tilelang-phase98.en.md)
> **优先级**: `107` · **问题 ID**: `flashinfer-grouped-fp8-silu-quant-lmhead-loss-dsv4-tmem-deepseek-route-cudnn-kv-tilelang-phase98` · **证据类型**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. 核心架构问题与结论速览

在 Blackwell（`sm_100a`/`sm_103a`）上部署大 $M$ MoE `gate_up` + SwiGLU + FP8 量化、分块 LM-Head 交叉熵损失、DSv4 BF16 H128 稀疏 MLA、DeepSeek 融合路由、cuDNN Paged Prefill 以及 TileLang Ascend/CUDA 编译降级时，Phase 98 通过 10 个零重叠一手上游提交揭示了以下核心工程规律：

1. **SM100/SM103 Grouped FP8 Fused SiLU-Quant Pair + Tail 核与每步原位重绑定 (`flashinfer#7454`, `1290367e8d`; `flashinfer#6049`, `492fb0128b`)**：大 $M$ MoE `gate_up` 路由在 CUDA Graph 下达到 `1.037x–1.154x` 提速（Eager 下 `1.050x–1.146x`），未变更的其余大 $M$ 形状保持 `0.984x–1.008x` 零退化；同时 `prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant` 支持在每次 `launch(a=, a_scale=, m_indices=, out_q=, out_s=)` 时直接重绑定每步激活张量而无需额外 staging 拷贝，并对 `m_indices == -1` 的填充行跳过写回，使单层只需持有一个 prepared 对象（32 层节省 `512 MiB/step` staging 流量）。
2. **分块 LM-Head Loss 单次设备同步行压缩与分尾 `dX` GEMM 波次 (`flashinfer#7449`, `c93782c532`)**：`chunked_lm_head_loss` 在前向与反向之间复用有效行索引（valid-row index），将每次调用的设备同步从 `2` 次降至 `1` 次，配合新 log-probability 核与分尾 `dX` GEMM 调度，在 GB300 上实现 `1.024x–1.068x` 加速，在 B200 上实现 `1.023x–1.051x` 加速。
3. **DSv4 BF16 H128 稀疏 MLA 的 TMEM 1/4 通道（Lane Quarter）统计量隔离 (`flashinfer#6895`, `c7f1e3dfe5`)**：Blackwell `tcgen05.ld` / `tcgen05.st` 要求每个 Warp 只能访问其所属的 TMEM 32-Lane 四分之一区。修复前 `meta_addr` 错误使用共享的 `head_row_base << 16`，导致多个 Warp 读写同一 TMEM 通道区；修复后改为 `score_row_base << 16`（`warp_idx * 32`），使 4 个 Warp 严格落在各自的 TMEM 1/4 通道内。
4. **DeepSeek 融合路由组索引上界构造保证 (`flashinfer#7111`, `6137d20476`)**：`cake_deepseek_fused_routing` 将 `group_index_candidate` 的初始值从 `0xFFFFFFFF`（`4294967295`）改为 `(unsigned int)(n_group - 1)`，从构造上保证即使输入全为 NaN 或退化分数，选出的组索引也严格满足 `< n_group`，彻底消除越界组寻址。
5. **Kimi K3 Latent MoE / Fused Router JIT 源码打包与 cuDNN Paged Prefill 松弛 KV 上界交集 (`flashinfer#7107`, `f810e74b5e`; `flashinfer#7382`, `395c50712c`)**：在 `pyproject.toml` 中将 `kimi_k3_latent_moe` 与 `kimi_k3_fused_router` 的 `csrc/**` 纳入 Wheel 分发；同时在 `cudnn_Helpers.py` 中通过 `graph_s_kv = min(graph_s_kv, block_tables.shape[1] * k_cache.shape[2])` 将调用方传入的宽松上界（如 `128`）收缩至物理页表容量（如 `3 * 16 = 48`），防止 cuDNN 图越界读取页表。
6. **TileLang Ascend DMA/切片基偏移保留、跨父循环周期缓冲生命周期与 CUDA 16-bit 数学内联 (`tilelang#3627`, `f5c6b44cfe`; `#3606`, `ed06ed1f01`; `#3172`, `43f13c7ad1`)**：Ascend `Copy::Make` 与 `MakeStridedSlice` 保留 `phys_buf.elem_offset + sum(min_i * stride_i)`（如 `256 + 132 = 388`）；`BufferAliasAnalyzerImpl` 仅在根级（`child == nullptr`）截断周期迭代分析，防止跨父循环复用的预加载 Buffer 被错误别名复用；CUDA 后端补齐 11 个 FP16/BF16 反三角与双目数学函数，并使用整数编码步进实现保留 BF16 次正规数（subnormals）的 `nextafter`。

---

## 2. 一手源码证据与定量指标对比

| 编号 | 证据 ID / 仓库提交 | 关键实现路径与硬件契约 | 核心定量收益 / 正确性不变量 |
| :--- | :--- | :--- | :--- |
| 1 | `flashinfer-cake-grouped-fp8-fused-silu-quant-pair-tail-7454-1290367-2026` (`1290367e8d`) | `cake_grouped_fp8_fused_silu_quant`: SM100/SM103 Pair+Tail 大 $M$ `gate_up` 核 | CUDA Graph `1.037x–1.154x`；Eager `1.050x–1.146x`；未改路由 `0.984x–1.008x` |
| 2 | `flashinfer-cake-grouped-fp8-gemm-per-call-operands-1-p-6049-492fb01-2026` (`492fb0128b`) | `grouped_fp8_gemm`: `launch(a, a_scale, m_indices, out_q, out_s)` + `-1` 跳过 | 0 次 staging 拷贝（省 `512 MiB/step`），`-1` 填充行零越界写 |
| 3 | `flashinfer-cake-lm-head-loss-log-probability-kernels-s-7449-c93782c-2026` (`c93782c532`) | `cake_lm_head_loss`: 复用前向有效行索引 + 分尾 `dX` GEMM | 同步次数 `2 -> 1`；GB300 `1.024x–1.068x`；B200 `1.023x–1.051x` |
| 4 | `flashinfer-cake-dsv4-keep-bf16-h128-softmax-stats-in-w-6895-c7f1e3d-2026` (`c7f1e3dfe5`) | `sparse_mla_fwd_bf16_h128.cu`: `score_row_base << 16` 替代 `head_row_base << 16` | 4 个 Warp 100% 落入各自 TMEM 32-Lane 1/4 通道，契合 `tcgen05.ld/st` |
| 5 | `flashinfer-cake-deepseek-fused-routing-bound-selected-7111-6137d20-2026` (`6137d20476`) | `routing_deepseek.cu`: `group_index_candidate = n_group - 1` 替代 `0xFFFFFFFF` | 构造上保证所选组索引 `0 <= idx < n_group`，消除 `4294967295` 越界 |
| 6 | `flashinfer-build-package-kimi-k3-jit-sources-7107-f810e74-2026` (`f810e74b5e`) | `pyproject.toml`: 纳入 `kimi_k3_latent_moe` 与 `kimi_k3_fused_router` `csrc/**` | Wheel 安装包可直接 JIT 编译 Kimi K3 Latent MoE 与融合路由 |
| 7 | `flashinfer-cudnn-accept-loose-kv-upper-bounds-in-paged-7382-395c507-2026` (`395c50712c`) | `cudnn_Helpers.py`: `min(graph_s_kv, block_tables.shape[1] * k_cache.shape[2])` | `min(128, 3 * 16) = 48` 截断宽松 `max_token_per_sequence`，消除页表越界 |
| 8 | `tilelang-ascend-preserve-buffer-elem-offset-in-dma-cop-3627-f5c6b44-2026` (`f5c6b44cfe`) | `ascend.cc`: `phys_buf.elem_offset + sum(min_i * stride_i)` | 切片偏移 `132 -> 388`（保留 `base=256`），修复子视图 DMA 错位 |
| 9 | `tilelang-ascend-preserve-complete-periodic-buffer-life-3606-ed06ed1-2026` (`ed06ed1f01`) | `memory_planning.cc`: 仅在 `child == nullptr` 时截断周期跨迭代分析 | 跨父循环迭代的预加载 Buffer 零误判重叠复用 |
| 10 | `tilelang-cuda-support-missing-16-bit-inverse-and-binar-3172-43f13c7-2026` (`43f13c7ad1`) | `half_math.h`: 11 个 16-bit 数学函数 + 整数步进 `nextafter` | 完整保留 BF16 subnormals（`0x0001 -> 0x0002`），避免 `__float2bfloat16` 冲零 |

---

## 3. 架构师设计练习与可证伪问题

1. **计算练习（TMEM Lane-Quarter 寻址与 `tcgen05` 硬件契约）**：在 Blackwell `sparse_mla_fwd_bf16_h128` 中，4 个 Warp 的 `score_row_base` 分别为 `0, 32, 64, 96`，而 `head_row_base` 均为 `0`。若 `meta_addr` 的高 16 位（Lane 偏移）使用 `head_row_base << 16`，4 个 Warp 访问的 TMEM Lane 区间分别是什么？为什么 `tcgen05.ld.sync.aligned.32x32b` 会触发未定义行为或跨通道数据覆盖？
2. **可证伪假设**：在 CUDA/HIP 上实现 `__nv_bfloat16` 的 `nextafter(x, y)` 时，能否先转为 `float32` 调用 `nextafterf` 再用 `__float2bfloat16` 转回？源码与 IEEE-754 位级实验证明：**不能**，因为 `nextafterf` 仅按 FP32 尾数（`2^-23`）步进 1 ULP，转回 BF16（`2^-7` 尾数）会被直接舍入回原值，且在次正规数区间会丢失精度；必须直接在 16-bit 整数编码上做 `±1` 步进。

---

## 4. 配套离线可复现实验

- **实验路径**: `experiments/flashinfer_grouped_fp8_silu_quant_lmhead_loss_dsv4_tmem_deepseek_route_cudnn_kv_tilelang_phase98/run.py`
- **校验命令**: `python experiments\flashinfer_grouped_fp8_silu_quant_lmhead_loss_dsv4_tmem_deepseek_route_cudnn_kv_tilelang_phase98\run.py --check`
- **单测覆盖**: `tests/test_phase98_experiments.py`
