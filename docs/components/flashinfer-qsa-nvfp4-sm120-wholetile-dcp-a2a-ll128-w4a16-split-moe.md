# FlashInfer SM120 QSA NVFP4 整块解码、SM100/103 DCP All-to-All LL128 延迟释放与 W4A16 Split MoE 架构审计

> **English Version**: [flashinfer-qsa-nvfp4-sm120-wholetile-dcp-a2a-ll128-w4a16-split-moe.en.md](flashinfer-qsa-nvfp4-sm120-wholetile-dcp-a2a-ll128-w4a16-split-moe.en.md)

## 1. 研究背景与核心问题

在 Blackwell 工作站/边缘服务器（SM120，如 NVIDIA RTX PRO 6000 Blackwell Server Edition，188 SM）与数据中心级 B200/GB300（SM100/SM103）的稀疏分页注意力（QSA）、上下文并行（DCP）与混合专家（MoE）推理中，三个核心硬件瓶颈决定了端到端延迟：
1. **SM120 QSA NVFP4 分页解码的 L2 污染与长流拆分问题**：当每行选择 2051 个稀疏 token（33 个 64-token tile）且 CTA 遍历较长 tile 流时，合作式融合（fused cooperative）解码在宽 batch（`rows >= 32`）下受限于 K-first 访存顺序和 L2 缓存驻留，吞吐量退化。
2. **SM100/SM103 DCP All-to-All LL128 FIFO 尾槽释放开销**：`decode_cp_a2a_alltoall` 在 CP2/CP4 跨 NVLink 交换部分注意力输出（BF16, `D=128`）和 softmax 统计量（2×FP32）时，若在当前 kernel 尾部同步释放最后一个 LL128 FIFO 槽位并通过共享内存写回，会引入额外的跨拍屏障延迟。
3. **W4A16 Split MoE 与 SM103 NVFP4 Attention 的 TMEM 布局权衡**：在 `cutlass_fused_moe(backend="cake")` 中引入 BF16 激活 × NVFP4 权重的双 kernel Split 形式（FC1 SwiGLU + FC2），以及在 SM103 NVFP4 Attention 中将 TMEM 从 432 扩至 480 列以驻留概率矩阵 `P`。

---

## 2. 第一手源码审计与硬件机制

### 2.1 SM120 QSA NVFP4 Whole-Tile Decode (`flashinfer-ai/flashinfer` @ `83755d9093`)
审计 `flashinfer/experimental/cake_qsa_nvfp4/cake_backend.py` 与 `README.md`：
- **Co-Resident 容量不变量**：在 SM90（H100, 132 SM）与 SM120（RTX PRO 6000, 188 SM）上，`FUSED_CTAS_PER_SM = 2`，因此 RTX PRO 6000 的合作式网格上限为 `188 × 2 = 376` CTAs（而 B200 为 148，GB300 为 152，R200 SM107 为 212）。
- **Whole-Tile 路由判据**：对于 `Hq=24, Hkv=2, D=256`，每行 2051 个选中 token 划分为 `ceil(2051 / 64) = 33` 个 64-token tile。SM120 新增 `decode_whole_tile`（仅提供 split 形式：decode + split-KV merge），在计算一个 64-token tile 前一次性发起该 tile 的全部异步拷贝，保持 2 级流水在途，并对拷贝施加 **L2 `evict-first`** 策略。
- **11-Tile (704 Tokens) 切换阈值**：当宿主端发现每个 CTA 遍历的 tile 数 `ceil(tiles / splits) >= 11`（即 `>= 704` 个选中 token，对应 `rows = 32/64/128` 时的 `3/2/1` splits）时，自动从 fused cooperative K-first 切换至 `decode_whole_tile + merge`：
  - Eager 模式下，`rows=32/64/128`（NHD）延迟从 `57.25 / 98.86 / 161.65 µs` 降至 `54.74 / 94.26 / 152.80 µs`（**1.05x–1.06x** 加速）；
  - CUDA Graph 回放模式下，延迟降至 `55.06 / 93.86 / 152.14 µs`（最高 **1.07x** 加速），而 `rows <= 16` 保持原融合路由不变（0.99x–1.01x）。

### 2.2 SM100/SM103 DCP All-to-All LL128 延迟释放 (`flashinfer-ai/flashinfer` @ `53feaae34c`)
审计 `csrc/cake_dcp_alltoall_dispatch.cu` 与生成的 `cake_dcp_alltoall_ff3dfded867e5738be23_kernel.cu`（CP2）/ `b8b8ebc8b91fd813f375`（CP4）：
- **跨启动槽位延迟释放（Deferred Last-Slot Release）**：接收端不再在当前 all-to-all 结束前等待并清空最后一个 LL128 FIFO 槽位，而是将该槽位的释放推迟到**下一次 kernel 启动的入口处**，同时将接收到的条目直接从寄存器以普通存储指令（plain stores）写入输出张量。
- **GB300 (`sm_103a`) 与 GB200 (`sm_100a`) 实测收益**：
  - GB200 CP2 `tokens=1`：Warm L2 从 `5.456 µs` 降至 `4.288 µs`（**1.27x**），Cold L2 从 `5.584 µs` 降至 `4.192 µs`（**1.33x**）；
  - GB300 CP4 `tokens=1..128`：全面提速 **1.16x–1.21x**（如 `tokens=1` 从 `5.440 µs` 降至 `4.480 µs`）。

### 2.3 Cake W4A16 Split MoE 与 SM103 NVFP4 Attention TMEM (`cf5c9a6a41`, `d08a1f7321`)
- **BF16 × NVFP4 W4A16 Split MoE (`cf5c9a6a41`)**：在 B300 (`sm_103a`) 上对比 FlashInfer CuTe DSL W4A16 基准，单卡 Preset D (`H=7168, I=2048, E=256, top-8`) 在 `T=1` 时从 `53.73 µs` 降至 `50.98 µs`（**1.054x**），8 卡 EP8 Preset G (`H=6144`) 在 `T=8` 达 **1.047x**（`58.63 -> 55.98 µs`），但在中等 batch `G128` 下因双 kernel 拆分开销回退至 `0.923x`（`123.31 vs 133.62 µs`），验证了按 token 档位动态选型 tactics 的必要性。
- **SM103 NVFP4 Attention TMEM 驻留 `P` (`d08a1f7321`)**：在 `cake_nvfp4_attention` 中将 `TMEM_NCOLS` 从 `432` 扩展到 `480`（新增 `TMEM_TMEM_P0_LO/HI_OFFSET = 448..472`），结合 tile 边界重叠，在 GB300 上实现 **1.02x–1.05x** 提速（`(4, 8, 4096)` 从 `0.1145 ms` 降至 `0.1094 ms`）。

---

## 3. 可复现实验与量化验证

运行本仓库原创模型 `experiments/flashinfer_qsa_nvfp4_sm120_wholetile_dcp_a2a_ll128_w4a16_split_moe/run.py --check`：

| 子系统 / 指标 | 基线配置 | Phase 81 优化路径 | 实测/验证结果 | 证据来源 |
|---|---|---|---|---|
| SM120 QSA NVFP4 (`rows=64`, Graph) | K-First Split (`100.35 µs`) | Whole-Tile + L2 Evict-First (`>=11` tiles/CTA) | **93.86 µs (1.07x)** | `src-flashinfer-cake-qsa-nvfp4-sm120-wholetile-2026` |
| GB200 DCP All-to-All (CP2, `T=1`) | 尾槽同步释放 (`5.584 µs` Cold) | 下次启动入口释放 + 寄存器直写 | **4.192 µs (1.33x)** | `src-flashinfer-cake-dcp-alltoall-sm100-sm103-2026` |
| B300 W4A16 Split MoE (`D1` / `G8`) | CuTe DSL W4A16 (`53.73 / 58.63 µs`) | Cake W4A16 Split (`50.98 / 55.98 µs`) | **1.054x / 1.047x** (`G128`: `0.923x`) | `src-flashinfer-cake-fused-moe-w4a16-split-2026` |
| GB300 NVFP4 Attention (`4×8×4096`) | `TMEM_NCOLS=432` (`0.1145 ms`) | `TMEM_NCOLS=480` + P-in-TMEM (`0.1094 ms`) | **1.05x** | `src-flashinfer-cake-nvfp4-attn-sm103-tmem-p-2026` |

---

## 4. 解决方案架构师设计建议與可证伪问题

1. **SM120 稀疏解码分流法则**：在 RTX PRO 6000 Blackwell (`376` 合作式 CTA 上限) 上，仅当单 CTA 处理 `>= 11` 个 64-token tile（`704` tokens）时启用 split `whole_tile` + L2 `evict-first`；短流或 `rows <= 16` 必须保留 fused cooperative K-first 以避免额外 merge kernel 启动开销。
2. **W4A16 Split MoE 混合路由**：由于 Cake W4A16 Split 在 `T=1..8` 领先而在 `G128` 落后 `7.7%`，生产环境不可全局静态替换 CuTe DSL W4A16，而应按 token bucket（`T <= 64` 选 Cake，`T in [96, 256]` 选 CuTe DSL）构建复合路由表。
