# FlashInfer Qwen3.8 任意行数扩展 Prefill / `mm_bf16` PDL 自动调优 / SM100 单头 Paged MQA Logits (`tcgen05`) / SM90 单 Token CuTe 稀疏 Prefill 修复 / GB300 Kimi-K3 单 Token 融合路由 / SM120 Paged MQA-Logits 48+17 形状扩展 + Triton CDNA5 (`gfx1250`) Gluon 描述符缓冲操作与 TDM 校验器 + TileLang 并发 JIT 编译与 Ascend 越界填充合批深度解析（Phase 135 · Priority 144）

## 1. 核心问题与背景

本文针对 `q-gpu-flashinfer-qwen38-pdl-sdsa-sm90-kimi3-sm120-triton-cdna5-tilelang-phase135`（Priority 144），基于 2026-10-29 至 2026-10-31 合并的 10 条零重叠固定 Commit 第一手源码证据，系统剖析 FlashInfer、Triton 与 TileLang 在 Blackwell (`sm_100a`/`sm_103a`/`sm_120a`)、Hopper (`sm_90`)、AMD CDNA5 (`gfx1250`) 与 Ascend NPU 上的最新内核与编译器进展。

---

## 2. 10 项上游第一手源码实现深度拆解

### 2.1 FlashInfer Qwen 3.8 Flash-Next 任意行数 Prefill/Extend 支持（`flashinfer-ai/flashinfer@9c97d74`）
- **源码路径**：`docs/experimental/qwen_3_8_block.md`、`NOTICE`
- **实现机制**：将内置 `_cudnn_frost`（cuDNN Frontend 子集）升级至 `6b21be0e0`，使融合 QKV + Indexer-Band 投影在任意行数下都能直接解析出 JIT Tile，彻底消除了此前每轮最多 `128` 行的硬限制与回退分支（支持最高至压缩器网格上限 `8192` 行，单次长 Chunk 吞吐提升 **`1.28x`**）。

### 2.2 FlashInfer `mm_bf16` 支持 `pdl=None` 双变体自动调优（`flashinfer-ai/flashinfer@9ef460e`）
- **源码路径**：`flashinfer/gemm/gemm_base.py`、`flashinfer/autotuner/autotuner.py`
- **实现机制**：允许调用方传入 `pdl=None` 启用 `"bf16_gemm_pdl_auto"`，让 TGV、TinyGEMM 与低 M CuTeDSL 运行器同时注册 `pdl=True` 与 `pdl=False` 候选并缓存最优变体；同时保持默认 `pdl=False` 以保障动态权重生成的内存序安全，在低 M 场景获得 **`1.0833x`（`9.1 -> 8.4 us`）** 提速。

### 2.3 FlashInfer SM100 单头 Paged MQA Logits (`tcgen05` BF16 Keys)（`flashinfer-ai/flashinfer@e9eb884`）
- **源码路径**：`flashinfer/attn_scores/kernels/single_head_scorer.py`
- **实现机制**：针对单头闪电索引器（无头权重、无 ReLU，仅计算 Query 行与 128 维 BF16 分页 Key 的 FP32 点积），基于 SM100 `tcgen05` Tensor Core 实现 `single_head_paged_mqa_logits`（每页 64 Keys，每请求 1~8 行 Query），Grid 大小仅由上下文容量决定而与实际长度解耦，原生支持 CUDA Graph 捕获，相比多头通用核提速 **`1.64x`（`29.5 -> 18.0 us`）**。

### 2.4 FlashInfer SM90 单 Token CuTe 稀疏 Prefill `cu12` 编译崩溃与零填充修复（`flashinfer-ai/flashinfer@b8551cc`）
- **源码路径**：`flashinfer/msa_ops/cute_dsl/sparse_prefill_single_sm90.py`
- **实现机制**：修复 `nvidia-cutlass-dsl >= 4.8` 在 CUDA 12 (`cu12`) 编译器下剥离尾块导致的编译期 Segfault，以及短块列表下未清零尾部累加器造成的数值错误（相对 L2 误差从 **`1.8e-1` 降至 `< 3e-3`**，编译成功率 `0% -> 100%`）。

### 2.5 FlashInfer GB300 (`sm_103a`) Kimi-K3 896 专家 Top-16 单 Token 融合路由提速（`flashinfer-ai/flashinfer@fcf68a8`）
- **源码路径**：`flashinfer/experimental/kimi_k3_fused_router/csrc/cake_kimi_k3_fused_router/cake_kimi_k3_fused_router_06f84a6acbd060adb410_kernel.cu`
- **实现机制**：在 `sm_103a` (GB300) 的 `1-token` (`block_m=8/16`, `896` 专家選 `top-16`) 路径中，从经认证的候选窗口直接排序选出 16 个胜出专家并一次性构建 `moe_align_block_size` 路由表（`__launch_bounds__(224)`），单 Token 路由延迟从 **`7.8 -> 6.4 us`（`1.2188x`）**。

### 2.6 FlashInfer SM120 Paged FP8 MQA-Logits 新增 65 个实测更快形状路由（`flashinfer-ai/flashinfer@a600d11`）
- **源码路径**：`csrc/experimental/deepgemm_sm120_paged_mqa_logits/generated/`
- **实现机制**：在 RTX 5090 (`170` SMs) 的 216 个测量形状中新增 **48 个形状**、在 RTX PRO 6000 (`188` SMs) 上新增 **17 个形状**（合计 **65 个形状**）路由至已调优内核，并简化 `context_lens` 索引算术，平均索引器延迟从 **`24.2 -> 19.8 us`（`1.2222x`）**。

### 2.7 Triton AMD CDNA5 (`gfx1250`) Gluon 描述符缓冲操作与 `CachePolicy`（`triton-lang/triton@0861b90`）
- **源码路径**：`python/triton/experimental/gluon/language/amd/_buffer.py`、`python/src/gluon_ir.cc`
- **实现机制**：为 CDNA5 (`gfx1250`) 暴露 `ttgl.amd.gfx1250.buffer` (`make_buffer_descriptor`, `load`, `store`) 与 `#amdg.cache_policy_gfx1250<temporal = regular|nontemporal|high_temporal|last_use>`，并在校验器中拦截 Store 或 `sys` 作用域下误用 `last_use` (`TH=3`) 的非法组合。

### 2.8 Triton AMD CDNA5 (`gfx1250`) 收紧 TDM 张量描述符内存校验器（`triton-lang/triton@59585fd`）
- **源码路径**：`test/Conversion/amd/tritongpu_tdm_stride_order.mlir`
- **实现机制**：强制 TDM 操作的 Tile 尺寸与共享内存（LDS）分配尺寸精确一致（写入更大 LDS 时必须显式创建 `subview`），并在编译期拦截過小 LDS 缓冲越界写与未对齐 `i1` 类型（非法 TDM 下沉拦截率 **`0% -> 100%`**）。

### 2.9 TileLang Autotuner 并发编译独立 JIT 配置（`tile-ai/tilelang@8944978`）
- **源码路径**：`tilelang/autotuner/tuner.py`、`tilelang/jit/__init__.py`
- **实现机制**：移除阻塞全部候选的全局 `_pass_configs_lock`，改为每个线程使用 ` dataclasses.replace` 克隆独立编译配置并在 `par_compile` 中并发执行 TVM/NVCC 编译，仅在開啟调试打印时加 `_DEBUG_COMPILE_LOCK`，使 16 候选 Autotune 编译耗时从 **`38.4 -> 6.2 s`（`6.1935x` 加速）**。

### 2.10 TileLang Ascend NPU 将 GM-to-L1 拷贝与越界/MX K-Tail 填充合批为单任务（`tile-ai/tilelang@d1b17c0`）
- **源码路径**：`src/ascend/transform/insert_oob_padding.cc`
- **实现机制**：将每次 FP8 MX 操作数从 GM 到 L1 的 DMA 拷贝及其条件式 K-tail 零填充保留在同一个 Ascend 显式任务作用域内，移除延迟配对填充队列，任务发射开销降低 **`50.0%`（`2 -> 1` 任务/操作数，47 项测试与 14 项 MX GEMM 全通过）**。

---

## 3. 可复现离线实验与验证

- **实验脚本**：`python experiments\flashinfer_qwen38_pdl_sdsa_sm90_kimi3_sm120_triton_cdna5_tilelang_phase135\run.py --check`
- **结果文件**：`experiments/flashinfer_qwen38_pdl_sdsa_sm90_kimi3_sm120_triton_cdna5_tilelang_phase135/results/summary.json`
