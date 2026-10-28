# FlashInfer MUSE Spark Norm SM107a、Grouped FP8 整行写回、MiniMax-H3 NVFP4、DSA 动态工单、GDN 确定性训练与 FA4/TileLang 精度验证（Phase 118 / Priority 127）

> **证据分类**：Upstream `implementation` + `author-benchmark`（10 个上游公开提交，2026-09-23 至 2026-10-28）+ 本仓库 `own-offline` 确定性算术/拓扑验证（`experiments/flashinfer_muse_spark_norm_fp8_gemm_minimax_dsa_gdn_fa4_tilelang_phase118/run.py`）。本文不包含未标注的真实 GPU 硬件重测声明。

## 1. 核心架构问题与背景

在 Blackwell (`SM100a`)、Blackwell Ultra (`SM103a`)、Rubin 过渡架构 (`SM107a` / `R200`) 及 GeForce Blackwell (`SM120a`) 的算子工程中，近期暴露了五类关键性能与数值瓶颈：

1. **SM107a 16-Head `rmsnorm_sigmoid_gate` 启动粒度过细与 PDL 窗口缺失**：旧版 `cake_muse_spark_norm` 在 16 头形状下使用 64 线程 16 行（单 Token）程序，未充分利用 `SM107a` 256-bit 向量通道，且在 128 头（9–24 tokens）与 32 头（10–12 tokens）缺少原生 `bf16_pdl` 程序而回退至 CuTe。
2. **Grouped FP8 GEMM 非对齐 BF16 输出的逐标量写回退化**：当 `group_gemm_fp8_nt_groupwise_contiguous` 的 BF16 输出张量仅满足 2 字节对齐（不满足 16 字节 TMA 对齐）时，宿主端走 `deepk_c2_scalar_output` 路由；旧版逐标量散写导致 B200/GB300 尾声延迟激增。
3. **DSA Indexer 固定分片长尾、MoE All-Reduce 跨卡冗余广播与 GDN 训练非确定性**：
   - `cake_dsa_indexer_topk` 在 `SM103a` 寬扫描下按静态集合分配工作单元，单个重流式单元会拖慢整个 Launch 的尾延迟；
   - `cake_moe_finalize_allreduce_fusion` 在 World Size 4/8 下采用 One-Shot 全广播，每 Rank 向所有 Peer 发送完整隐藏层分片，导致大 Token 数下 NVLink 流量膨胀；
   - Gated DeltaNet (`chunk_gated_delta_rule`) 训练若依赖原子累加或数据相关调度则丧失逐位确定性，且 FlashAttention-4 SM100 稀疏 MLA 反向在计算 $dS = P \circ (dP - dP_{\text{sum}}) \cdot \text{scale}$ 前进行了三次 BF16 截断。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与微架构机制 |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@b08e86326a` (`#7505`) | 2026-10-28 | `flashinfer/cake_muse_spark_norm_backend/`：统一 `ROW_ALIGNMENT_BYTES = 32`（`GATE_STRIDE_ELEMENTS = 16`），新增 `sm_107a` 128 线程多 Token 程序（单 CTA 容纳 4 个 16-head Token，每线程 2 行 256-bit 访问）及小 Token `bf16_pdl` 尾无关程序。R200 (`sm_107a`, 212 SMs) 实测 `(16, 16384)` 达 **1.5366x / 1.5616x** 加速。 |
| 2 | `flashinfer-ai/flashinfer@ccb6fc8d4c` (`#7597`) | 2026-10-28 | `csrc/cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_3be71ae2bb86939e1cd8_kernel.cu`：为 `sm_100a`/`sm_103a` 的 `deepk_c2_scalar_output` 路由新增整行写回程序（26 条路由背后扩展至 28 个程序）。B200 实测 2 字节偏移输出在 `correctness_unaligned_output` 从 `21.95 us` 降至 `5.36 us`（**4.10x**），`ldpz_mixed_m301` 达 **2.08x**。 |
| 3 | `flashinfer-ai/flashinfer@574d6d8d53` (`#7599`) | 2026-10-28 | `csrc/cake_minimax_h3_sm120_nvfp4_varlen_attention/`：升级 `SM120` MiniMax-H3 `[tokens, 56, 128]` NVFP4 变长注意力，TMA 描述符开启 `allow_oob_box=(false, true, false)` 自动零填充尾 Tile，插入 `fence.proxy.async.shared::cta`，并用 `thread_local uint64_t prepared_devices` 位图缓存 `49536 B` 动态共享内存设置。 |
| 4 | `flashinfer-ai/flashinfer@b532fa4443` (`#7571`) | 2026-10-28 | `flashinfer/experimental/cake_dsa_indexer/`：`sm_103a` 宽扫描程序不再静态绑定工作单元，而是从 Workspace 末尾 `TICKET_TAIL_BYTES = 128` 字节中的无符号 32 位工单字动态领取下一单元；等长分段采用升序块序，不等长保持最长优先。GB300 上 `Z1k4096` 从 `12.701 ms` 降至 `11.285 ms`（**1.13x**）。 |
| 5 | `flashinfer-ai/flashinfer@3cf451835d` (`#7511`) | 2026-10-28 | `csrc/cake_moe_finalize_allreduce_fusion/`：在 `H=7168` 下为 WS4（`>=96` tokens）与 WS8（`>64` tokens）引入同一 Lamport 缓冲区内的 Two-Shot（Reduce-Scatter + 顺序保序求和 + All-Gather）程序，并为 WS2 引入片上保留自身部分和的 `own-partial` 大 Token 程序。720 个形状几何平均加速 **1.121x**，WS8 2048 tokens 在 B300/B200 上达 **2.902x / 2.874x**（最高 **3.468x**）。 |
| 6 | `flashinfer-ai/flashinfer@6b6c9f996d` (`#7592`) | 2026-10-28 | `flashinfer/experimental/cake_gdn_det_train/`：新增 `SM100`/`SM103` 确定性 Gated DeltaNet 分块训练后端（`d=128, chunk=64`），全程无原子操作与数据相关乱序；在 `HV = 2H` 时融合反向尾部内核。相比 FLA 0.5.2，B200 上加速 **1.184x–3.587x**（`1x16384` 从 `2.9256 ms` 降至 `0.9421 ms`，**3.106x**）。 |
| 7 | `flashinfer-ai/flashinfer@ee76d93ae8` (`#7595`) | 2026-10-28 | `flashinfer/experimental/nvfp4_mla_decode/`：长行解码引入基于交换屏障投票门控（Vote-Gated Max Exchange）的 `decode_xv`（预热开销 `XV_WARMUP_TILES = 48`，稳态门槛 `XV_MIN_STEADY_TILES = 307`），短单元引入按 Warp 选路写回的 `decode_short_epi_warp`（末尾 CTA 有效行 `>= 32`）。 |
| 8 | `flashinfer-ai/flashinfer@94090279f3` (`#4559`) | 2026-10-27 | `flashinfer/fused_moe/`：在 W4A16 MoE 缩放因子归约中消除大型中间临时张量，直接在线归约每组缩放因子（在 `[64, 4096, 56]` FP16 规模下节省 `28.0 MiB` 显存分配）。 |
| 9 | `Dao-AILab/flash-attention@ce1d8313d5` | 2026-09-23 | `flash_attn/cute/flash_bwd_mla_sm100.py`：修复 SM100 稀疏 MLA 反向 Softmax Warp 将 `scale_p`、归一化 $P$ 与 $(dP - dP_{\text{sum}}) \cdot \text{scale}$ 提前舍入为 BF16 的三次截断问题，全程保留 FP32 $P^T$ 并在 $dS$ 写回前仅执行 1 次 BF16 舍入。 |
| 10 | `tile-ai/tilelang@af08dae6e4` (`#3465`) | 2026-10-28 | `src/backend/common/op/atomic_simt.h`：抽取 CUDA 与 ROCm 共享的 SIMT 原子操作迭代与索引映射校验器，在编译期拦截源/目标 Region 维度阶数（Rank）或对应轴范围（Extent）失配并抛出 `ValueError`。 |

---

## 3. 关键公式、数据移动与离线验证结果

### 3.1 MoE Finalize + All-Reduce：One-Shot vs Two-Shot NVLink 流量模型
设节点内并行度为 $W$，隐藏维度为 $H = 7168$，元素字节数为 $b = 2\text{ B}$：
- **One-Shot 全广播**：每个 Rank 将本地部分和直接推送给其余 $W - 1$ 个 Rank，每 Token 跨 NVLink 总传输字节数为：
  $$V_{\text{1-shot}} = W(W - 1)Hb$$
- **Two-Shot（Reduce-Scatter + All-Gather）**：每个 Rank 将 $1/W$ 分片发给对应 Owner Rank（共 $W(W-1)\frac{Hb}{W} = (W-1)Hb$），Owner 按 Rank 顺序確定性累加后再广播回其余 $W-1$ 个 Rank（再加 $(W-1)Hb$），每 Token 跨 NVLink 总传输字节数为：
  $$V_{\text{2-shot}} = 2(W - 1)Hb$$
当 $W = 8$ 时，$\frac{V_{\text{2-shot}}}{V_{\text{1-shot}}} = \frac{2}{W} = 0.25$，即跨卡 NVLink 流量降低 **75%**（`moe_ar_ws8_nvlink_traffic_reduction_ratio = 0.75`），解释了 B300/B200 在 2048 tokens 下高达 **2.902x / 2.874x** 的实测加速。

### 3.2 离线模型验证汇总（`run.py --check`）

- `norm_row_alignment_bytes`: `32`（`sm_107a` 16-head 多 Token 程序在 `(16, 16384)` 达 `1.5367x` 加速）
- `grouped_fp8_gemm_unaligned_b200_speedup`: `4.1`（`deepk_c2_scalar_output` 整行写回在 B200 上实现 `4.10x` 加速，`ldpz_mixed_m301` 达 `2.08x`）
- `minimax_dyn_smem_bytes`: `49536`（每设备 `thread_local` 位图缓存设置 + TMA 轴 1 `allow_oob_box` 尾块零填充）
- `dsa_ticket_tail_bytes`: `128`（GB300 `Z1k4096` 动态工单加速 `1.1255x` / 报告值 `1.13x`）
- `moe_ar_ws8_nvlink_traffic_reduction_ratio`: `0.75`（WS8 Two-Shot 降低 75% NVLink 字节数，2048 tokens 实测 `2.902x`）
- `gdn_det_train_16k_b200_speedup`: `3.105`（`1x16384 H8/HV16` 确定性训练从 `2.9256 ms` 降至 `0.9421 ms`）
- `nvfp4_mla_xv_min_total_tiles`: `355`（`48` 预热 Tile + `307` 稳态 Tile 门限）
- `fa4_sparse_mla_bwd_roundings_saved`: `2`（消除 $dS$ 生成前的 2 次多余 BF16 舍入）
- `tilelang_simt_invalid_atomic_shapes_rejected`: `2`（拦截 Rank 失配与 Extent 失配两类非法 SIMT 原子映射）

---

## 4. 面向云解决方案架构师的练习与可证伪问题

1. **计算练习**：在 `W = 8, H = 7168, BF16 (2 B)` 的单节点 MoE Finalize + All-Reduce 中，当 Batch Token 数 $T = 2048$ 时，计算 One-Shot 与 Two-Shot 方案在单次内核调用中穿越 NVLink 的总数据量（以 MiB 为单位）。（答案：One-Shot 为 $2048 \times 8 \times 7 \times 7168 \times 2 / 2^{20} = 1568\text{ MiB}$；Two-Shot 为 $2048 \times 2 \times 7 \times 7168 \times 2 / 2^{20} = 392\text{ MiB}$，单次调用节省 $1176\text{ MiB}$ 跨卡流量）。
2. **可证伪假设**：在 `cake_nvfp4_mla_decode` 中，为何不能对所有序列长度无条件启用 `decode_xv`（投票门控最大值交换）？因为基于屏障投票的门控路径在每个分片（Piece）启动时需支付约 `XV_WARMUP_TILES = 48` 个流水线 Tile 的预热代价；只有当单集群在预热之后还需流式处理至少 `XV_MIN_STEADY_TILES = 307` 个 Tile（总计 $\ge 355$ 个 Tile）时，省下的跨 Warp 最大值交换开销才能抵消预热成本。
