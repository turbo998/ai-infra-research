# FlashInfer XQA E4M3x2 解码转换、SM107 1024 线程/SM 量化填充、CAKE Grouped MXFP8/FP8 尾块移交与 TileLang 动态 cp.async 守卫（Phase 117 / Priority 126）

> **证据分类**：Upstream `implementation`（10 个上游公开提交，2026-10-27）+ 本仓库 `own-offline` 确定性算术/拓扑验证（`experiments/flashinfer_xqa_sm107_cake_mxfp8_gemm_vscale_kda_tilelang_phase117/run.py`）。本文不包含未标注的真实 GPU 硬件重测声明。

## 1. 核心架构问题与背景

在 Blackwell Ultra / Rubin 过渡代（`SM100a`/`SM103a`/`SM107`/`SM120`）以及 AMD `gfx950` 异构编译栈中，算子微架构面临四类典型瓶颈：

1. **XQA BF16-Query 解码中的逐字节 E4M3 转换指令膨胀**：在 `sm_89+` 上，当 Query 为 BF16 而 KV Cache 为 FP8 E4M3（或 NVFP4 的 E4M3 块缩放因子）时，旧版 XQA 每次仅对单个字节执行 `e4m3x2 -> f16x2` 转换，再转 `f32 -> bf16`（RTZ），白白浪费一半 SIMD 转换通道。
2. **SM107 硬件线程上限减半导致的带宽饥饿**：`SM107`（208–212 SMs）每 SM 最大常驻线程数为 **1024**（`max_threads_per_multi_processor = 1024`），而非常规 Blackwell `GB300` 的 `2048`。按 2048 线程/SM 设计的量化与 `fused_rmsnorm_silu` 启动形状在 SM107 上在途字节数（bytes in flight）直接腰斩，无法打满 HBM 流式带宽上限。
3. **CAKE Grouped MXFP8 量化尾块空转与 Grouped FP8 GEMM 跨 Warpgroup 尾声同步**：分组 MoE 在 `K=4096` 下的末尾稀疏专家常产生不满 128 行的局部 Tile；同时 `group_gemm_fp8_nt_groupwise_contiguous` 的 KG1 向量输出程序若跨 Warpgroup 共享 staging 面板，会引入多余的 epilogue barrier。
4. **多模态双向注意力 V-Scale Swizzle 失配、GLM-5.3 KDA 头数覆盖与 TileLang 动态 `cp.async` 等待深度**：`trtllm-gen` 写入的分页 NVFP4 KV Cache 对 V 缩放因子采用了页内 swizzle，导致 Gemma 3/4 图像前缀双向范围预填充无法直接读取；TileLang 在动态循环排空阶段生成非常量 `1 - ko` 等待深度时触发 `tirx.Sub -> ir.IntImm` 转型崩溃。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与微架构机制 |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@0e44b0ef80` (`#6762`) | 2026-10-27 | `csrc/xqa/utils.cuh`：复用双字节 `e4m3x2 -> f16x2` 转换路径后接 `f32 -> bf16` RTZ，保证包括 `NaN` 与 `-0` 在内的逐位等价，在 SM120 上消减约 **16%** 的 BF16-query FP8 KV 解码总指令数（转换指令减半 `50%`）。 |
| 2 | `flashinfer-ai/flashinfer@b2a10d6062` (`#6207`) | 2026-10-27 | `csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh` & `flashinfer/norm/__init__.py`：针对 `SM107`（1024 threads/SM）倍增每线程在途字节数并重调 CTA 启动网格，使 `quantize` 与 `fused_rmsnorm_silu` 在途字节恢复 `2.0x`。 |
| 3 | `flashinfer-ai/flashinfer@bd8fea0abb` (`#7615`) | 2026-10-27 | `csrc/cake_grouped_mxfp8_quantize/*_kernel.cu`：`K=4096`（256 线程）per-K 行分块程序将尾部残余 Group 的局部 Tile 移交给前序有效 Group 的活跃 CTA 处理，消除空 Group CTA 空转波次（示例负载下节省 `65.62%` CTA 波次）。 |
| 4 | `flashinfer-ai/flashinfer@9c20838062` (`#7611`) | 2026-10-27 | `csrc/cake_grouped_fp8_gemm/*_kernel.cu`：在 27 个程序（14 条 FP32-scale 路由 + 12 条 UE8M0 路由）中的 KG1 向量输出程序（`12f65758`/`a14c6ac9`）内，让每个 Warpgroup 独立暂存并写回自身面板，消除跨 Warpgroup 尾声屏障（尾声停顿降低 `38.1%`）。 |
| 5 | `flashinfer-ai/flashinfer@a3aae554a9` (`#7012`) | 2026-10-27 | `flashinfer/attention/_core.py` & `flashinfer/jit/attention/modules.py`：`BatchPrefillWithCausalBidirectionalRangesWrapper` 支持直接按页反交织读取 `trtllm-gen` 写入的 Swizzled NVFP4 V 缩放因子，打通 Gemma 3/4 视觉前缀与 `trtllm-gen` 解码共享同一 KV Cache。 |
| 6 | `flashinfer-ai/flashinfer@ecb0464fcb` (`#7510`) | 2026-10-27 | `flashinfer/fused_moe/trtllm_gen_routing.py`：在 `_TRTLLM_GEN_ROUTING_SUPPORTED_CC` 中补齐 `107`（支持 `[100, 103, 107, 120, 121]`），解除 SM107 独立 MoE 路由拦截。 |
| 7 | `flashinfer-ai/flashinfer@36ac484d7c` (`#7554`) | 2026-10-27 | `flashinfer/experimental/kimi_k3_fp8_projection/cake_backend.py`：为 Kimi-K3 TP 本地 FP8 投影更新第 12 轮路由表（`SM100a`/`SM103a`），在 `quant + persistent 2-CTA tcgen05 GEMM`、`quant + decode GEMM` 与 `fused in-CTA quant decode GEMM` 三种形态间按 `(N, K, M)` 分桶派发。 |
| 8 | `flashinfer-ai/flashinfer@f9ad55c831` (`#7132`) | 2026-10-27 | `flashinfer/kda_kernels/fused_kda_decode.py`：`fused_kda_decode` 与 `packed_fused_kda_decode` 允许列表新增 `16` 与 `64` 头（完整覆盖 GLM-5.3-Flash 的 TP1=64、TP2=32、TP4=16、TP8=8），并原生支持 vLLM 维度优先 DS 卷积状态布局 `[slots, 3*H*128, 3]`。 |
| 9 | `tile-ai/tilelang@6551d2ce7d` (`#3547`) | 2026-10-27 | `src/cuda/codegen/codegen_cuda.cc` & `src/rocm/codegen/codegen_hip.cc`：当软件流水线排空阶段生成非常量 `cp_async` 等待深度时，自动安全回退为立即数全等待（PTX `cp.async.wait_group 0` / AMDGPU `s_waitcnt vmcnt(0)`），修复编译期 `IntImm` 强转崩溃。 |
| 10 | `tile-ai/tilelang@e530c6d159` (`#3621`) | 2026-10-27 | `src/rocm/codegen/intrin_rule_hip.cc`：修复 HIP 后端默认将 `T.exp`/`T.sin`/`T.cos` 降级为低精度 fastmath 内建函数的缺陷，默认保持 IEEE 精度并将显式快速算子导出至 `FAST_MATH_INTRINSICS`。 |

---

## 3. 关键公式、数据布局与离线验证结果

### 3.1 SM107 在途字节数与 Little 定律填充
设单 SM 最大常驻线程数为 $T_{	ext{SM}}$，每线程单波次发起的向量加载字节数为 $B_{	ext{thr}}$，则单 SM 最大并发在途字节数为：
$$B_{	ext{in-flight}} = T_{	ext{SM}} 	imes B_{	ext{thr}}$$
在 `GB300` 上 $T_{	ext{SM}} = 2048$，而在 `SM107` 上 $T_{	ext{SM}} = 1024$。若沿用 $B_{	ext{thr}} = 16	ext{ B}$，$B_{	ext{in-flight}}$ 仅为 $16384	ext{ B/SM}$；`#6207` 通过将每线程在途字节提升至 $32	ext{ B}$，使 `SM107` 恢复 $32768	ext{ B/SM}$（`sm107_bandwidth_fill_gain_ratio = 2.0`）。

### 3.2 离线模型验证汇总（`run.py --check`）

- `xqa_cvt_inst_reduction_ratio`: `0.5`（单头 E4M3 转换指令从 `128` 降至 `64`，对应 SM120 端到端解码指令减少 `16.0%`）
- `sm107_bandwidth_fill_gain_ratio`: `2.0`（单 SM 在途字节从 `16384 B` 恢复至 `32768 B`）
- `cake_mxfp8_cta_wave_saving_ratio`: `0.6562`（8 Group 偏斜负载下有效 CTA 数从 `32` 压缩至 `11`）
- `cake_fp8_epilogue_stall_reduction_ratio`: `0.381`（独立 Warpgroup 面板暂存消除了跨 Warpgroup 尾声等待）
- `all_glm53_tp_covered`: `true`（`[8, 12, 16, 24, 32, 48, 64, 96]` 完整覆盖 GLM-5.3-Flash TP1..TP8）
- `fallback_wait_group_immediate`: `0`（动态 `cp.async` 排空深度安全降级为 `wait_group 0` / `vmcnt(0)`）

---

## 4. 面向云解决方案架构师的练习与可证伪问题

1. **计算练习**：对于 `image_tokens = 2048, kv_heads = 8, head_dim = 256` 的 NVFP4 双向视觉前缀（每 16 个 FP4 元素携带 1 字节 E4M3 缩放因子），计算在 Prefill 内核中直接反交织读取 V-Scale 相比独立 Deswizzle 内核节省的 HBM 读写字节数。（答案：单份 V-Scale 体积为 $2048 	imes 8 	imes (256/16) = 262,144	ext{ B}$，省去独立一读一写共节省 $524,288	ext{ B}$）。
2. **可证伪假设**：在 `SM107` 上将每线程在途字节数翻倍是否会因寄存器压力跨过 64 regs/thread 阈值而反噬占用率？由于 `SM107` 每 SM 具有 65536 个 32-bit 寄存器且常驻线程上限仅为 1024，即使每线程使用 64 个寄存器仍可保持 100% 线程上限（$1024 	imes 64 = 65536$），这正是 `SM107` 与 `2048 threads/SM` 架构的关键差异。
