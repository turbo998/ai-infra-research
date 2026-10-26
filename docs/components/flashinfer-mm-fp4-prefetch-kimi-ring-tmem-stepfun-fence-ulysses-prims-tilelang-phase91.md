# Phase 91 Track A：FlashInfer Cake NVFP4 L2 权重预取、Kimi-K3 环形槽位与宽 MLA TMEM 擦除、Blackwell 异步代理栅栏、Ulysses NVLink 直写、PrimTS MLA 跳过重缩放与 TileLang 标量 ABI 解析

> **English Version**: [flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91.en.md](flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91.en.md)  
> **阶段状态**：Phase 91（`priority: 100 — 100-Priority Milestone!`）· 关联问题 `q-flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91` · 关联论断 `claim-flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91`  
> **可复现实验**：[`experiments/flashinfer_mm_fp4_prefetch_kimi_ring_tmem_stepfun_fence_ulysses_prims_tilelang_phase91/`](../../experiments/flashinfer_mm_fp4_prefetch_kimi_ring_tmem_stepfun_fence_ulysses_prims_tilelang_phase91/)

---

## 1. 一手源码证据矩阵（10 项上游提交）

| # | 仓库与提交 | 日期 | 证据级别 | 核心架构机制 |
|---|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer` [`fc933c5bdd`](https://github.com/flashinfer-ai/flashinfer/commit/fc933c5bdd11292c30a87db38e44f1ae4dc74984) (`#7424`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_mm_fp4` 在 PDL 网格依赖等待前将前 `CHAIN_EP_STAGES = 4` 个 K-tile 的 NVFP4 权重与 block-16 缩放因子流式预取入 L2（B200 64 行加速 `1.010x–1.076x`，GB300 56 行加速 `1.012x–1.081x`） |
| 2 | `flashinfer-ai/flashinfer` [`6785a485ac`](https://github.com/flashinfer-ai/flashinfer/commit/6785a485acaa6da113672da819bd99eb99e2cf18) (`#6900`) | 2026-10-26 | `implementation` | `cake_kimi_k3_fp8_projection` 驻留 decode 槽位改用 `(res_stage_m + 3u) % 4u`，消除第 4 个 token tile 时 `res_stage_m == 0` 导致的无符号下溢 `0xFFFFFFFF` |
| 3 | `flashinfer-ai/flashinfer` [`4bd511ac3d`](https://github.com/flashinfer-ai/flashinfer/commit/4bd511ac3dbde1851a80fa9d8f4df7315ef83750) (`#6899`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_kimi_k3_mla` 2-CTA 宽程序 TMEM 擦除将 `scrub_row` 统一固定为 `o_tmem_row_base_1`，满足 `tcgen05.st` 全 lane 统一基地址契约 |
| 4 | `flashinfer-ai/flashinfer` [`9cce37835f`](https://github.com/flashinfer-ai/flashinfer/commit/9cce37835fc8cb38c09409335c36c0f82b88bb6f) (`#7422`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_kimi_k3_vision_tower`（27 层 MoonViT-3D + PatchMergerV2）在 `T >= 8192` 引入 `l_e8_tma1x` / `l_e8_tma1x_h` TMA-epilogue 2-CTA 对（B200 最高 `1.074x`，B300 最高 `1.058x`） |
| 5 | `flashinfer-ai/flashinfer` [`6bd679e419`](https://github.com/flashinfer-ai/flashinfer/commit/6bd679e419ff52aa6eb382f84a70a59429a3b642) (`#7428`) | 2026-10-26 | `implementation` | `cake_stepfun_moe` MXFP8 tile-256 FC1 在 `cp.async.wait_group 1` 与 `tcgen05.cp` 之间插入 `fence.proxy.async.shared::cta` |
| 6 | `flashinfer-ai/flashinfer` [`05e425835b`](https://github.com/flashinfer-ai/flashinfer/commit/05e425835b8d9777b01972d10a2083da6b5e92aa) (`#7437`) | 2026-10-26 | `implementation` | `cake_minimax_h3` SM120 CTA-epilogue 程序在 MMA warp 释放主循环阶段前执行代理栅栏，防止 TMA 覆盖共享显存 |
| 7 | `flashinfer-ai/flashinfer` [`dc50addf12`](https://github.com/flashinfer-ai/flashinfer/commit/dc50addf1244fd614fa50477179f788369221d1a) (`#7440`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_ulysses` 新增 24 个 NVLink 对称 IPC 注册输出直写程序，消除 staging buffer 拷贝（B300 8 卡 world=6 scatter `1.108x`、world=8 gather `1.094x`） |
| 8 | `flashinfer-ai/flashinfer` [`396ffbad72`](https://github.com/flashinfer-ai/flashinfer/commit/396ffbad72133e3d364efdf91b25f10cb4844522) (`#7170`) | 2026-10-26 | `implementation` | `cake_kda` BF16 BT16 prepare 在通用代理清零尾块后、下一 chunk 异步 TMA 装载前插入 `fence.proxy.async.shared::cta` |
| 9 | `flashinfer-ai/flashinfer` [`6ba6bdf377`](https://github.com/flashinfer-ai/flashinfer/commit/6ba6bdf377ff65a5233c7c89b55fbbf1f8e2e655) (`#7026`) | 2026-10-26 | `implementation` | PrimTS 稠密/稀疏 MLA decode 新增 `MLAAttentionTuning.softmax_rescale_threshold_log2`（E4M3 最高 `8` binades、`P_scale = 448 * 2**-T`；BF16 最高 `32` binades） |
| 10 | `tile-ai/tilelang` [`f2b8866e73`](https://github.com/tile-ai/tilelang/commit/f2b8866e733f667dd273ac383fbe30fab2098d1e) (`#3602`) | 2026-10-26 | `implementation` | TileLang JIT 通过参数位置映射 `get_output_scalar_dim_map` 解析动态标量输出维度，不向 Cython/PTO 内核 ABI 重复追加标量参数 |

---

## 2. 关键实现路径与架构深挖

### 2.1 PDL 依赖等待期的 NVFP4 权重 L2 预取（`#7424`）
在 Blackwell SM100/SM103 的 `cake_mm_fp4` per-token 量化 + GEMM 链中，下游 GEMM 通过 Programmatic Dependent Launch（PDL）紧跟上游激活量化 kernel 启动。当激活量化尚未完成时，GEMM CTA 阻塞在 `cudaGridDependencySynchronize()`。`#7424` 利用这段等待窗口，让 GEMM 中原本空闲的预取 warp 先把前 `CHAIN_EP_STAGES = 4` 个 K-tile（每个 `128×128` tile 含 `8,192 B` NVFP4 权重 + `1,024 B` block-16 缩放因子，共 `9,216 B`，4 级合计 `36,864 B/CTA`）流式预取入 L2，使网格依赖解除后第一波 TMA 加载直接命中 L2。

### 2.2 PTX 通用代理与异步代理显存序不变式（`#7428`, `#7437`, `#7170`）
Blackwell / Hopper 共享显存在 PTX 内存模型下区分 **generic proxy**（普通 `st.shared` / `cp.async`）与 **async proxy**（TMA `cp.async.bulk`、`tcgen05.cp`、`tcgen05.mma`）：
- **StepFun MoE MXFP8（`#7428`）**：`cp.async` 通过 generic proxy 写入共享显存中的 B 缩放因子，随后 leader CTA 调用 `tcgen05.cp` 通过 async proxy 读入 TMEM；仅靠 `cp.async.wait_group 1` 不建立跨代理可见性，必须插入 `fence.proxy.async.shared::cta`。
- **KDA BF16 BT16（`#7170`）**：尾块修复用普通 `st.shared` 清零 `smem_q_raw / smem_k_raw / smem_gate_raw`，两个 `__syncthreads()` 后下一 chunk 用 TMA 覆写同一缓冲区；若不在 `__syncthreads()` 前执行 `fence.proxy.async.shared::cta`，旧通用写可能晚于新 TMA 写落盘。

### 2.3 Ulysses NVLink 对称 IPC 直写与 PrimTS MLA 跳过重缩放（`#7440`, `#7026`）
- **Ulysses 直写路由（`#7440`）**：当输出由 `UlyssesCommunicator.allocate_output` 分配并通过 CUDA IPC 对称映射时，scatter/gather kernel 直接将分片写入远端 GPU 的目标输出张量，省去中间 staging buffer 往返，在 B300 8-GPU 上取得 `1.045x–1.108x` 加速。
- **PrimTS MLA 跳过在线 softmax 重缩放（`#7026`）**：当块间最大值增幅未超过阈值 $T = \text{softmax\_rescale\_threshold\_log2}$ 时跳过累加器重缩放；对于 FP8 E4M3，为防止未重缩放的 $P = \exp(S - m_{\text{old}})$ 溢出 $448$，将 $P$ 预缩放因子设为 $448 \times 2^{-T}$（$T \in [0, 8]$）。

---

## 3. 架构师设计练习与可证伪问题

1. **计算题**：在 `cake_mm_fp4` 中，若 CTA tile 为 $M \times N \times K = 64 \times 128 \times 128$，预取 `CHAIN_EP_STAGES = 4` 个 K-tile 的 NVFP4 权重与 block-16 FP8 缩放因子，148 个活跃 CTA 合计预取多少 KiB 数据进入 L2？
   - **解答**：单 K-tile 权重 $128 \times 128 \times 0.5\text{ B} = 8,192\text{ B}$，block-16 缩放因子 $128 \times (128/16) \times 1\text{ B} = 1,024\text{ B}$，合计 $9,216\text{ B} = 9\text{ KiB}$。4 个 stage 为 $36\text{ KiB/CTA}$，148 个 CTA 合计 $148 \times 36\text{ KiB} = 5,328\text{ KiB} \approx 5.20\text{ MiB}$，远小于 B200/GB300 的 L2 容量。
2. **可证伪假设**：若在 `cake_kimi_k3_fp8_projection` 中开启单 CTA 驻留 4 个 token tile 的调度表项而不包含 `#6900`，第 4 个 tile 的 `cur_slot` 将解引用 `res_stage_m - 1 == 0xFFFFFFFF` 引发越界访问。
