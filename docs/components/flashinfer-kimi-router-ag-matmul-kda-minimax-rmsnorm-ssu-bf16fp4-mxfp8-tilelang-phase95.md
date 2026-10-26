# FlashInfer Kimi-K3 融合路由、All-Gather Matmul Producer-Tail、KDA 双 CTA CP 扫描、MiniMax-H3 NVFP4、RMSNorm 分阶段 dWeight、SSU P-Split、Blackwell BF16×FP4 与 TileLang SM120 展开深挖 (Phase 95 / Priority 104)

> **证据分类 (Evidence Class)**：上游源码实现 (`implementation`) + 本项目 Python 3.12 标准库离线确定性模型 (`own-offline`)。本文不宣称真实 GPU 硬件重测。

## 1. 核心架构问题与一手源码落点

| 信源 ID | 仓库与提交 | 关键实现路径与架构不变量 |
|---|---|---|
| `flashinfer-kimi-k3-fused-router-sm100a-pr7441-2026` | `flashinfer-ai/flashinfer` @ `bdd721538c` (PR #7441) | `kimi_k3_fused_router`（896 专家、top-16、FP32 logits+bias、`block_m` $\in \{8, 16\}$、28 个测量网格）更新 `sm_100a`/`sm_103a` 路由程序：1 token 采用闭式单 CTA 构造；2..16 tokens（2 的幂）采用 `num_tokens` CTA 集群通过分布式共享内存（DSMEM）交换 top-16 ID；32-token cell 采用 16-CTA 集群服务至多 32 行；256-token cell 显式采用 1-CTA cluster；4-CTA 集群按设备共驻容量（`CAPACITY_ROUTES`）路由。 |
| `flashinfer-allgather-matmul-peer-cta-release-wait-pr6885-2026` | `flashinfer-ai/flashinfer` @ `0af9594727` (PR #6885) | 修复 `cake_all_gather_matmul` 7 个 `cta_group::2` 主程序的退出竞态：Leader CTA 的 MMA warp 通过多播 `tcgen05.commit` 向两个 CTA 的 `mma_done` mbarrier 释放 6 级 TMA 环的每一级。Peer CTA 的 load warp 原本只在 refill 前等待 `mma_done`，最后一次 refill 后直接退出会导致在途异步 arrive 落入已退出 CTA 的共享内存（违反 PTX ISA §2.2.2，且 PTX §9.7.18.6 不保证 `tcgen05.commit` 间顺序）。修复后在 tile 循环末尾追加 6 级 `producer_tail` 等待并翻转 `load_phase`。 |
| `flashinfer-cake-kda-2cta-per-launch-cp-state-walk-pr7436-2026` | `flashinfer-ai/flashinfer` @ `4f14b25dc6` (PR #7436) | `cake_kda_train` 上下文并行（CP）变体（`head_dim=128, chunk=64`，支持 contiguous / zigzag / per-document zigzag+ragged 三种切分与 FP32 仿射状态摘要跨卡传递）为 4 个片段状态递推扫描（前向/反向 × offset/tape）新增按 launch 选择的 2-CTA 记录：每个 `(segment, head)` 由 2 个 CTA 各负责 64 列状态。 |
| `flashinfer-minimax-h3-nvfp4-pre-attention-sm100-pr7434-2026` | `flashinfer-ai/flashinfer` @ `9e656abf02` (PR #7434) | `MiniMaxH3Nvfp4PreAttention` 交付 `sm_100a`/`sm_103a` × `nvfp4`/`e4m3` 全部 4 个变体（Stage 1: RMSNorm+AdaLN+NVFP4 量化；Stage 2: NVFP4 QKV GEMM + Q/K RMSNorm + NeoX RoPE + destination-major pack 融合尾声），作者在 GB300 (`sm_103a`, 1400W) 报告加速比 `1.0257x–1.1198x` (`nvfp4`) 与 `1.0274x–1.1140x` (`e4m3`)，GB200 达 `1.0010x–1.0603x`。 |
| `flashinfer-cake-rmsnorm-staged-dweight-chunk-taper-pr7417-2026` | `flashinfer-ai/flashinfer` @ `a8d23b20bf` (PR #7417) | `cake_rmsnorm_train` 更新 `sm_100a`/`sm_103a`/`sm_107a` 反向算子对（plain 与 residual-fused），引入分阶段 `dweight` 归约与 chunk taper（`max_halving=5`, `tail_div=12`, `rows_per_issue=4`），在 `sm_107a` ($H=2048$) 作者报告最高 `1.13x–1.18x` 加速。 |
| `flashinfer-ssu-psplit-tma-ring-persistent-decode-pr7407-2026` | `flashinfer-ai/flashinfer` @ `fdd121f46b` (PR #7407) | `selective_state_update(..., backend="cake")` 在 `sm_100a`/`sm_103a` 上引入 P-split（将每个 128 行程序头沿 head 维切分为 2 或 4 个 CTA，每 CTA 64 或 32 行以填满 SM）、TMA-ring 与持久化 decode 程序。 |
| `flashinfer-blackwell-bf16-fp4-tcgen05-small-m-pr7419-2026` | `flashinfer-ai/flashinfer` @ `328e862b34` (PR #7419) | `mm_bf16_fp4` 重新生成 97 个程序中的 10 个：3 个双 CTA $M=128$ `tcgen05` 程序在最后一次存储仍在途时提前通过 `tmem_dealloc_bar_addr` 释放 Tensor Memory（TMEM），消除末尾全集群屏障等待；7 个 `blackwell-tiled` 小 $M$ 寄存器流式 split-K 程序在主循环前一次性加载并转换全部块缩放因子。 |
| `flashinfer-grouped-mxfp8-quant-row-tiled-k7168-pr7431-2026` | `flashinfer-ai/flashinfer` @ `bdaa2401ad` (PR #7431) | `cake_grouped_mxfp8_quantize` 新增按 $K$ 选择的行分块程序（当输入 $\ge 4\text{ MiB}$ 且 $B \cdot M \ge 16384$ 行时触发）：$K=4096$ 采用 256 线程 CTA（每 CTA 2 行），$K=2048$ 采用 128 线程 CTA，$K=7168$（224 线程）改用整数运算计算 UE8M0 缩放字节替代 FP32 除法，且主机端全程不读取 `mask[B]`。 |
| `tilelang-out-idx-caller-launch-stream-alloc-pr3649-2026` | `tile-ai/tilelang` @ `89b5b5b9ad` (PR #3649) | 修复 `cython`、`nvrtc`、`cutedsl` 适配器跨流内存覆写缺陷：当调用方传入非当前流的 `stream=` 时，在 `output_allocation_stream` 上分配 `out_idx` 输出张量，避免 PyTorch 缓存分配器复用当前流上仍有未完成核函数写入的内存块。 |
| `tilelang-sm120-block-scaled-rega-gemm-k-unroll-pr3286-2026` | `tile-ai/tilelang` @ `32f56122e2` (PR #3286) | SM120 块缩放寄存器 A (`is_gemm_rs`) GEMM 在 TIR 中显式展开 K-atom 循环 (`T.unroll(explicit=True)`)，消除跨多个 64-wide K-atom 时的运行时索引与局部内存（local memory）寄存器溢出；同时每 lane 每 K-atom 仅加载 $\lceil \text{warp\_rows}/2 \rceil$ 个 SFA 字与 $\lceil \text{n8\_blocks}/4 \rceil$ 个 SFB 字，替代每对 MMA 3 次共享内存加载（在 $\text{warp\_rows}=2, \text{n8\_blocks}=8$ 下从 24 次降至 3 次，缩减 `8.0x`）。 |

## 2. 架构师计算与设计练习

1. **All-Gather Matmul `cta_group::2` Producer-Tail 同步练习**：设 TMA 加载环深度 $D=6$，总 tile 数为 $T=15$。验证主循环结束时 `load_stage = 15 mod 6 = 3`、`load_phase = (15 // 6) mod 2 = 0`；执行 6 次 `#pragma unroll 1` 尾部 `mbarrier_wait_token` 后，`load_stage` 回到 `3` 且 `load_phase` 翻转为 `1`，恰好排空全部 6 个在途多播 `tcgen05.commit` 到达信号。
2. **SM120 块缩放 GEMM 共享内存缩放因子读取缩减计算**：当每个 warp 覆盖 $\text{warp\_rows}=2$、$\text{n8\_blocks}=8$ 时，每个 64-wide K-atom 包含 $2 \times (8/2) = 8$ 个 MMA 对。旧路径每 MMA 对执行 3 次共享内存读取（共 24 次/lane）；新路径每 lane 预装 $\lceil 2/2 \rceil + \lceil 8/4 \rceil = 1 + 2 = 3$ 个紧凑选择字并通过线程 ID 操作数索引，共享内存缩放读取降低 `8.0x`。

## 3. 可复现实验与后续可证伪问题

- **实验命令**：`python experiments/flashinfer_kimi_router_ag_matmul_kda_minimax_rmsnorm_ssu_bf16fp4_mxfp8_tilelang_phase95/run.py --check`
- **后续可证伪问题**：在 $B \cdot M < 16384$ 但总字节数 $> 4\text{ MiB}$（例如超宽 $K=16384$）的分组 MXFP8 量化场景下，256 线程双行 CTA 是否仍优于 128 线程单行 CTA？
