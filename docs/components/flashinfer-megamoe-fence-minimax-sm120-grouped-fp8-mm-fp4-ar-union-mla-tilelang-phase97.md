# FlashInfer MegaMoE EP16 `fence.proxy.alias`、MiniMax-H3 SM120 Out-Proj、Grouped FP8 Direct-Store、MM-FP4 Ring-Overlay、MoE AR Union、NVFP4 MLA QK-Ahead 与 TileLang 编译器演进（Phase 97）

> **English Version**: [flashinfer-megamoe-fence-minimax-sm120-grouped-fp8-mm-fp4-ar-union-mla-tilelang-phase97.en.md](flashinfer-megamoe-fence-minimax-sm120-grouped-fp8-mm-fp4-ar-union-mla-tilelang-phase97.en.md)
> **优先级**: `106` · **问题 ID**: `flashinfer-megamoe-fence-minimax-sm120-grouped-fp8-mm-fp4-ar-union-mla-tilelang-phase97` · **证据类型**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. 核心架构问题与结论速览

在 Blackwell（`sm_100a`/`sm_103a`/`sm_120a`）上推进 16-Rank MegaMoE 全融合算子、分组 FP8/NVFP4 GEMM、MLA Decode 以及 TileLang CUDA/TIR 编译器降级时，本阶段（Phase 97）通过 10 个零重叠上游提交揭示了以下关键架构规律：

1. **跨 Rank 虚拟地址别名同步 (`fence.proxy.alias`) 必须先于 `fence.acq_rel.sys` 与信号到达 (`flashinfer#7378`, `364b53a4bf`)**：在 `CakeMxfp8MegaMoeEp16`（16 Ranks、512 专家 = 32/Rank、Top-8、`H=3072, I=5120`）中，各 Rank 通过 Peer-Mapped VA（`remote_mc_va`）写入路由与隐藏态数据，再由接收端通过本地 VA 读取。若将 `fence.proxy.alias` 放在 `fence.acq_rel.sys` 之后或屏障等待之后，远端写入与本地别名读取不在同一同步路径上保序；将 `fence.proxy.alias` 移至 `fence.acq_rel.sys` 紧前方可消除跨 Rank 别名读取竞态。
2. **RTX PRO 6000 Blackwell (`sm_120a`, 188 SMs) 专属 Out-Proj 尾声分块与 16B 成对缩放加载 (`flashinfer#7389`, `b986d13848`)**：`cake_minimax_h3_sm120_quant_out_proj` 新增 `PROGRAM_1_SMS = 188` 路由，每个 MMA Warp 在私有 SMEM 块中暂存尾声切片，并以单次 16-Byte SMEM 加载读取每个阶段对的 NVFP4 缩放行，在 `M=33472` 上将 FP8 耗时从 `3705.5 us` 降至 `3630.0 us`（`1.021x`），NVFP4 提速 `1.040x–1.046x`。
3. **Grouped FP8 GEMM 的 FP32 缩放 Direct-Store 与 256 行对齐 2-CTA 派发 (`flashinfer#7386`, `63b0744f22`)**：新增 `deepk_c2_kg1_direct_store_m32`、`deepk_c2_ab6_scale1_direct_store_m32`（`M <= 32` 且 `CTAs <= 16`）与 `deepk_c2_kg1_expert_aligned`（Tile 不跨双专家边界），并为 2-CTA 路由引入 256 行对齐检查。
4. **SM100 `cake_mm_fp4` 主循环 SMEM 环形缓冲尾声复用 (`flashinfer#7380`, `3508ca8255`)**：当 `M <= 128` 且 `ceil(N / 128) > num_sms`（例如 B200 148 SMs 下 `N=28672` 产生 `224 > 148` 个 128 宽 Tile）时，将 `128x128` 1-CTA 输出暂存叠加在主循环 SMEM 环上，使流水线级数从 `5` 级扩至 `6` 级，在 `K=8192, N=28672, M=128` 上获得 `1.019x–1.024x` 提速。
5. **MoE AllReduce Union 交错两阶段 Reduce-Scatter 与首 Token 短链加载 (`flashinfer#7387`, `5a052b767f`)**：`cake_trtllm_moe_allreduce_union`（`H=7168`）缩短单/双 Token 首块加载链并为 `WS=8, M in {64, 128}` 引入交错 Two-Shot Reduce-Scatter，在 B200 上全 30 个形状提速 `1.02x–1.11x`（`WS=2, M=1`: `7.87 -> 7.49 us`, `1.0507x`），在 B300 上提速 `1.01x–1.12x`。
6. **NVFP4 MLA Decode 双 Tile QK-Ahead 流水与 Split-KV 合并提速 (`flashinfer#7385`, `aa3d76f9ba`)**：当 `unit_tiles >= 512`（`QK_AHEAD_MIN_UNIT_TILES = 512`）时，`decode_qk_ahead` 让 QK MMA 领先 PV 两个 KV Tile 发射，配合 `BALANCED_MIN_KV = 8192` 的 Split-KV 合并核，在 B200 64K/128K 上提速 `1.039x/1.055x`，GB300 上 Split-2 达 `11.97 -> 11.78 us`（`1.016x–1.026x`）。
7. **Blackwell `mm_bf16_fp4` 移除尾随 Cluster Barrier (`flashinfer#7376`, `de80f27971`)**：移除 6 个 `(1,1,4)` 与 `(1,1,8)` Split-K 核末尾冗余的 `barrier.cluster.arrive/wait`，在 `1x4096x4096` 上提速 `1.075x–1.078x`，在 `16x1024x1024` Leader-Mailbox 上提速 `1.139x–1.140x`。
8. **TileLang 寄存器内 BF16x2→FP32x2 转换、传递拷贝依赖与批处理规约修复 (`tilelang#3595/#2822/#3070`)**：`__tl_cvt_bf16x2_to_f32x2` 用 `shl.b32 %0, %2, 16; and.b32 %1, %2, 0xffff0000;` 替代需要字节重排的 `__bfloat1622float2`；`inject_pipeline` 统一标量与 Buffer 反向工作表以保留传递拷贝等待；`FinalizeReducerLowerer` 通过 `effective_batch > 1 && layout_batch_size > 0` 与线性偏移修复批处理 Warp 规约。

---

## 2. 一手源码证据与定量指标对比

| 编号 | 证据 ID / 仓库提交 | 关键实现路径与硬件契约 | 核心定量收益 / 正确性不变量 |
| :--- | :--- | :--- | :--- |
| 1 | `flashinfer-cake-megamoe-ep16-alias-proxy-fence-pr7378-2026` (`364b53a4bf`) | `cake_mxfp8_megamoe_ep16`: `fence.proxy.alias` 紧邻 `fence.acq_rel.sys` 之前 | 16 Ranks / 512 Experts (`H=3072, I=5120`) 跨 VA 别名顺序写读 100% 保序 |
| 2 | `flashinfer-cake-minimax-h3-sm120-outproj-rtxpro6000-pr7389-2026` (`b986d13848`) | `cake_minimax_h3_sm120_quant_out_proj`: `PROGRAM_1_SMS=188` + 16B scale | FP8 `M=33472`: `3705.5 -> 3630.0 us` (`1.0208x`)；NVFP4 达 `1.046x` |
| 3 | `flashinfer-cake-grouped-fp8-gemm-direct-store-fp32-scale-pr7390-2026` (`63b0744f22`) | `cake_grouped_fp8_gemm`: 新增 3 个 FP32-scale 核与 14 路形状派发 | `M<=32, CTAs<=16` 直写输出，单专家对齐免双专家分支，256 行 2-CTA 对齐 |
| 4 | `flashinfer-cake-mm-fp4-ring-overlay-128wide-sm100-pr7380-2026` (`3508ca8255`) | `cake_mm_fp4`: `128x128` 1-CTA 主循环 SMEM 环形叠加尾声 | 主循环流水从 `5 -> 6` 级，`K=8192, N=28672, M=128` 达 `1.019x–1.024x` |
| 5 | `flashinfer-cake-trtllm-moe-allreduce-union-interleaved-2shot-pr7384-2026` (`5a052b767f`) | `cake_trtllm_moe_allreduce_union`: 交错 Two-Shot RS (`H=7168`) | B200 全 30 形状 `1.02x–1.11x` (`WS=2, M=1`: `7.87 -> 7.49 us`)；B300 达 `1.12x` |
| 6 | `flashinfer-cake-nvfp4-mla-decode-qkahead-split-kv-pr7385-2026` (`aa3d76f9ba`) | `cake_nvfp4_mla_decode`: `QK_AHEAD_MIN_UNIT_TILES=512` (2-Tile Ahead) | B200 64K/128K 提速 `1.039x/1.055x`；GB300 Split-KV 合并提速 `1.026x` |
| 7 | `flashinfer-cake-blackwell-bf16-fp4-splitk-cluster-pr7376-2026` (`de80f27971`) | `mm_bf16_fp4`: 移除 6 个 Split-K 核末尾冗余 Cluster Barrier | B200 `1/4096/4096` 达 `1.078x`；`16/1024/1024` Leader-Mailbox 达 `1.140x` |
| 8 | `tilelang-cuda-vectorized-bf16-to-fp32-conversion-pr3595-2026` (`1a52a1cd06`) | `__tl_cvt_bf16x2_to_f32x2`: `shl.b32 16` + `and.b32 0xffff0000` | 32-bit 寄存器内直接展开高低 BF16 通道，消除 `prmt` 重排开销 |
| 9 | `tilelang-preserve-transitive-pipeline-copy-deps-pr2822-2026` (`4b2032a6ce`) | `inject_pipeline.cc`: 标量+Buffer 统一反向依赖工作表 + 线程域扩展 | 跨级传递 Producer-Consumer 异步拷贝等待零遗漏 |
| 10 | `tilelang-fix-batched-finalize-reducer-lowering-pr3070-2026` (`6db1fc771a`) | `lower_shared_barrier.cc`: `layout_batch_size > 0` 线性偏移批处理规约 | 多 Batch `tl.finalize_reducer` 每个 Batch 独立写入 `b * layout_batch_size` |

---

## 3. 架构师设计练习与可证伪问题

1. **计算练习（SM100 Ring-Overlay 级数与波次）**：在 B200（`148` SMs）上运行 `M=128, N=28672, K=8192` 的 FP4 GEMM，若采用 `128x256` Tile（`112` 个输出 Tile < `148` SMs），SM 利用率为 `112/148 = 75.7%`；改用 `128x128` Ring-Overlay Tile（`224` 个输出 Tile）并复用主循环 SMEM 后，为什么能同时将流水线从 `5` 级提升至 `6` 级并实现 `1.024x` 加速？
2. **可证伪假设**：在多节点/单节点多 GPU NVLink 映射中，若内核同时通过多播/对等别名 VA 写入并通过本地单播 VA 读取，仅执行 `fence.acq_rel.sys` 而不执行 `fence.proxy.alias` 是否足以保证本地读取看到最新写入？实验与 PTX 内存模型证明：**不能**，异构虚拟地址别名必须经过 `fence.proxy.alias` 同步。

---

## 4. 配套离线可复现实验

- **实验路径**: `experiments/flashinfer_megamoe_fence_minimax_sm120_grouped_fp8_mm_fp4_ar_union_mla_tilelang_phase97/run.py`
- **校验命令**: `python experiments\flashinfer_megamoe_fence_minimax_sm120_grouped_fp8_mm_fp4_ar_union_mla_tilelang_phase97\run.py --check`
- **单测覆盖**: `tests/test_phase97_experiments.py`
