# FlashInfer Cake Warp Decode / Rubin `sm_107a` Mega-MHC FP8 Group-128 / SM120 MiniMax-H3 最长优先排空 / Ulysses 8-Way 2-Unit 直散 / Top-K 流级 `row_states` 缓存 / NVFP4 预提倒数 / MXFP8 PDL 早启 + Triton CGA DAG 切片 / CDNA5 硬件 `V_TANH` + TileLang 单轮跨 Warp AllReduce 深度解析（Phase 134 · Priority 143）

## 1. 核心问题与背景

本文针对 `q-gpu-flashinfer-cake-mhc-ulysses-topk-nvfp4-mxfp8-triton-cga-cdna5-tilelang-phase134`（Priority 143），基于 2026-10-30 合并的 10 条零重叠固定 Commit 第一手源码证据，系统剖析 Blackwell (`sm_100a`/`sm_103a`/`sm_120a`)、Rubin (`sm_107a`) 与 AMD CDNA5 (`gfx1250`) 算子栈的关键优化。

---

## 2. 10 项上游第一手源码实现深度拆解

### 2.1 FlashInfer Cake Warp Decode `sm_100a` 双行启动配置（`flashinfer-ai/flashinfer@36ec610`）
- **源码路径**：`csrc/fused_moe/warp_decode/generated/cake_warp_decode_generated_manifest.cuh`
- **实现机制**：
  1. `e128_i1536_k8` 在 `8` Tokens 时切换至预等待认领（Pre-Wait Claim）兄弟 FC2 核（在等待 FC1 屏障前先领取任务项），在 GB200 Cold-L2 memset 刷新模式下从 **`107.28 -> 106.67 us`（`1.0057x`）**。
  2. `h3072_e256_i768_k8` 在 `3` Tokens 时将融合 route-pack FC1 任务池上限设为 `128` 槽位，Grid 从 `12 x 12`（`144` CTAs）收敛为 `12 x 10`（`120` CTAs，削减 `16.67%` 空转 CTA），Read-Flush 延迟从 **`22.48 -> 22.12 us`（`1.0163x`）**。

### 2.2 FlashInfer Rubin `sm_107a` `cake_deepgemm_mega_mhc` FP8 Group-128 原始缩放提速（`flashinfer-ai/flashinfer@7832229`）
- **源码路径**：`csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_ed3f407e5a68eb65478e_kernel.cu`
- **实现机制**：针对 `sf_layout="fp8_group128"` 原始缩放（Raw Scale）模式，使用 `rcp.approx.ftz.f32` + 2 条 `fma.rn.f32` 牛顿迭代与范围谓词 `scale in [8.6736e-19, 1.1529e+18)` 替代慢速 IEEE `div.rn.f32`，使尾阶段量化吞吐提升 **`1.0526x`（`38.0 -> 36.1 us`）**，主干 mHC 程序集增至 `37` 个。

### 2.3 FlashInfer Cake MiniMax-H3 `sm_120a` NVFP4 变长注意力两类尾部最长优先排空（`flashinfer-ai/flashinfer@386c7c7`）
- **源码路径**：`flashinfer/diffusion_ops/cake_minimax_h3_sm120_nvfp4_varlen_attention.py`
- **实现机制**：当变长分段 Key-Tile 种类数 `<= _HEAD_MAJOR_TAIL_CLASSES (2)` 时，对最后 `_HEAD_MAJOR_TAIL_ROUNDS (4) * grid` 个工作单元按**最长段优先（Longest-First）**排序排空，避免长 Tile 在最后 1 波孤立执行造成 SM 尾部饥饿，将 `seg3_ragged` 延迟从 **`148.0 -> 132.5 us`（`1.1170x`）**。

### 2.4 FlashInfer Cake Ulysses World-8 BF16 Direct Scatter 2-Unit 单轮拷贝（`flashinfer-ai/flashinfer@10cfb36`）
- **源码路径**：`flashinfer/cake_ulysses_backend/csrc/cake_ulysses_a2a_0f6daa3f5ad400c9b220_kernel.cu`
- **实现机制**：当对端 Rank 尚未进入核函数时，发送方将首趟前 **2 个 16-Byte 单元（Slots 0 & 1）** 暂存至信号区尾部（信号缓冲大小保持 `1,138,688` 个 32-bit Words 不变），接收方通过 CTA 单次投票后由 Warps `1..15` 在 **1 轮（原 2 轮）** 内拷入注册输出，World-8 Direct Scatter 延迟从 **`21.4 -> 18.5 us`（`1.1568x`）**。

### 2.5 FlashInfer Multi-CTA Radix Top-K 流级 `row_states` 缓存（`flashinfer-ai/flashinfer@6336f3e`）
- **源码路径**：`flashinfer/topk.py` (`_get_row_states_buffer`)
- **实现机制**：内核退出前会自行清零被污染的 `row_states` 槽位，且单 CTA 路径完全不触碰该缓冲。通过 `_get_stream_cache_buf` 按 CUDA Stream 缓存 `1 MiB` (`1,048,576 B`) 缓冲，彻底消除 9 个生产算子每次调用的 `torch.zeros` 分配与清零开销（单次发射开销从 **`4.8 -> 1.2 us`，`4.0x` 加速**）。

### 2.6 FlashInfer NVFP4 量化在零缩放检查前预提全局倒数（`flashinfer-ai/flashinfer@1d2cd26`）
- **源码路径**：`csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh`
- **实现机制**：在 E4M3 转换与零缩放检查前提前发射独立的 `reciprocal_approximate_ftz(SFScaleVal)`，隐藏 MUFU 指令延迟，同时严格保持非零块四舍五入为 0 时的 CuTe 零缩放契约，寄存器保持 `72` 个不变，量化核提速 **`1.0417x`（`7.50 -> 7.20 us`）**。

### 2.7 FlashInfer Swizzled MXFP8 量化器 PDL 早启消费者（`flashinfer-ai/flashinfer@80ece84`）
- **源码路径**：`flashinfer/quantization/kernels/mxfp8_quantize.py`
- **实现机制**：在 `griddepcontrol_wait()` 之后立即调用 `cute.arch.griddepcontrol_launch_dependents()`，让下游 CuTe-DSL GEMM 的序言初始化与上游量化计算重叠。在 GB200 (`M=4, N=6144, K=1024`, Split-K=4, CUDA Graph) 实测延迟从 **`8.559 -> 8.175 us`（降低 `4.51%`，`1.0470x`）**。

### 2.8 Triton `AssignCGALayouts` DAG 切片消除冗余 `convert_layout`（`triton-lang/triton@e9d175a`）
- **源码路径**：`lib/Dialect/TritonNvidiaGPU/Transforms/AssignCGALayouts.cpp`
- **实现机制**：将 CGA 布局传播从树形切片改为 DAG 切片（使用 `MapVector` 与 `DenseSet`），当同一个 `tt.descriptor_load` 喂给多个 `tt.dot` 时直接将目标 `CGALayout` 赋给生产者，消除 `2 -> 0` 次冗余 `ttg.convert_layout`。

### 2.9 Triton AMD CDNA5 (`gfx1250`) 硬件 `V_TANH` 与精确 `tanh` 分离（`triton-lang/triton@8abae58`）
- **源码路径**：`third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp`
- **实现机制**：将 `libdevice.fast_tanhf` 在 CDNA5 上下沉为单条硬件指令 `llvm.amdgcn.tanh.f32`（`8 -> 1` 条指令，**`8.0x` 指令压缩**），同时移除 `libdevice.tanh` 的硬件拦截，保持精确 `tanh` 使用 OCML 软件实现。

### 2.10 TileLang CUDA 跨 Warp AllReduce 折叠为单轮共享内存交换（`tile-ai/tilelang@0176ba1`）
- **源码路径**：`src/tl_templates/cuda/reduce.h`
- **实现机制**：先执行 Warp 内归约，再将所有跨 Warp 层级折叠为 **1 轮共享内存交换 + 2 次屏障**（原 256 线程块需 3 轮共享内存交换与 6 次屏障）。在 B300 (`sm_103a`) `per_block_cast` 全 8 种配置上，`BAR.SYNC` 从 **`12 -> 4`（`-66.7%`）**、`STS` 从 **`6 -> 2`（`-66.7%`）**，寄存器零增长。

---

## 3. 可复现离线实验与验证

- **实验脚本**：`python experiments\flashinfer_cake_mhc_ulysses_topk_nvfp4_mxfp8_triton_cga_cdna5_tilelang_phase134\run.py --check`
- **结果文件**：`experiments/flashinfer_cake_mhc_ulysses_topk_nvfp4_mxfp8_triton_cga_cdna5_tilelang_phase134/results/summary.json`
