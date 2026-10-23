# FlashInfer SM107a Hidden-4096 Fused mHC Quantized Outputs, Frost Single TVM-FFI Entry & TileLang Pipeline Transitive Dependencies

> **Language**: [中文](flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline.md) | [English](flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline.en.md)
> **Phase**: 65 (`priority: 74`) · **Track A**: GPU Kernels, Low-Bit Quantization & Compiler Passes
> **Verified Upstream Sources**: `flashinfer-ai/flashinfer@21ac136220` (`#7114`), `flashinfer-ai/flashinfer@c0e51ae29b` (`#7127`), `flashinfer-ai/flashinfer@b90521314b` (`#7037`), `tile-ai/tilelang@4b2032a6ce` (`#2822`), `tile-ai/tilelang@1bafce1a37` (`#3581`), `tile-ai/tilelang@d9a31ddd1c` (`#3579`)

---

## 1. 核心架构问题与研究背景

在 Blackwell (`sm_100a` / `sm_103a` / `sm_107a`) 推理架构与新一代 DSL 编译器（CuTe-DSL / TileLang）演进中，算子执行效率不再仅受限于 Tensor Core 算力，而高度取决于 **片上量化融合（Epilogue Quantization Fusion）**、**Host 侧 FFI 发射开销（Host Submission Overhead）**、**中等 Token 批次的 Tile 粒度空洞**，以及 **异步流水线编译器的依赖分析完备性**：

1. **SM107a Hidden-4096 Fused mHC 算子的二次 HBM 读取瓶颈 (`flashinfer@21ac136220`)**：此前 `flashinfer.mega_mhc` 在 `sm_107a` (Hidden=4096) 上仅提供 `sf_layout="bf16"` 变体。当下游紧接 FP8 Group-128 或 NVFP4 算子时，必须先将 `[T, 4096]` BF16 激活写回 HBM，再调用 vLLM 的 `per_token_group_quant_fp8` 或 `scaled_fp4_quant` 重新从 HBM 读回，浪费 `8 KiB/token` 的额外 HBM 读带宽。
2. **Frost 低延迟 MoE 的 Python/CuTe 多重 FFI 边界开销 (`flashinfer@c0e51ae29b`)**：在 B300 上执行 $T \in \{1, 2, 4\}$ 的 Decode MoE 时，GPU kernel 本身仅耗时 `10.33–26.65 µs`，但 Warm 态的 Python Runtime 每次调用仍需重新构造 DLPack 包装器、指针描述符，并分 4 次跨越 Python/CuTe FFI 边界（`pack_scale`、`gate_up`、`down_project`、`combine`），导致 Host 侧 CPU 耗时高达 `83.25–260.85 µs`，完全掩盖了底层硬件加速收益。
3. **Cake StepFun Fused MoE 在 $T=1024$ 处的 MXFP8 Tile 空洞 (`flashinfer@b90521314b`)**：StepFun 64-Expert Top-8 SwiGLU MoE (`H=4096, I=1536`) 的 MXFP8 算子库此前仅包含 Tile `{8, 16, 32, 64, 256}`，缺少 `tile=128`。在 $T=1024$ 时，默认调度器被迫回退到 `tile=64`（CTA 数量翻倍、权重重复加载）或 `tile=256`（专家 Token 填充浪费严重）。
4. **TileLang 软件流水线的传递拷贝依赖丢失与跨线程 Region 抵消 (`tilelang@4b2032a6ce` & `1bafce1a37`)**：
   - 在 `pipeline_planning.cc` 中，当 Stage 1 的间接全局内存拷贝地址依赖于 Stage 0 写入 Shared Memory 的索引缓冲（如 `A[idx_smem[i]]`）时，仅追踪直接数据流会遗漏传递拷贝依赖（Transitive Copy Dependency），导致异步 TMA/cp.async `wait` 被提前移除。
   - 在判断两个 BufferRegion 是否冲突时，若直接做符号区间相减，共享的线程变量 `threadIdx.x` 会在两阶段间错误抵消（忽略了写操作在线程 $t_1$、读操作在线程 $t_2$ 的跨线程 RAW/WAR 冲突）。
   - 同时，Ascend PTO 后端的 `PTOMmadDirectionAnalyzer` 在遇到含可变内存读取的分支时构造 `S || S`，32 层分支会导致谓词 AST 呈 $2^{32}$ 指数级膨胀，引发 10 分钟以上的编译卡死 (`tilelang@1bafce1a37`)。

---

## 2. 源码级实现机制与公式推导

### 2.1 FlashInfer Cake Mega-mHC Hidden-4096 量化输出融合 (`21ac136220`)

在 `flashinfer/experimental/deepgemm_mega_mhc/mega_mhc.py` 中，新增了两个 `sm_107a` Hidden-4096 导出程序（`cake_mega_mhc_a940c94296afad2b2497` 与 `cake_mega_mhc_1c61fed11caf93d1b762`），使 `sf_layout` 支持三态路由：`("bf16", "fp8_group128", "nvfp4")`：

- **Bitwise-Identical 基础输出不变量**：无论选择哪种 `sf_layout`，算子写出的四个基础张量——`new_residual` (`BF16 [T, 4096]`)、`y_bf16` (`BF16 [T, 4096]`)、`new_post_mix` (`FP32`) 与 `new_comb_res_mix` (`FP32`)——与原生 BF16 算子保持 **逐比特完全一致（Bitwise Identical）**。
- **`sf_layout="fp8_group128"` 契约**：在寄存器内对 `y_bf16` 的每 128 列分组计算缩放因子，直接写出 `y_fp8` (`FP8-E4M3 [T, 4096]`) 与 `y_fp8_sf` (`FP32 [T, 32]`)。当 `fp8_sf_ue8m0=True`（默认）时，缩放因子严格对齐为 2 的整数次幂：
  $$s_{t,g}^{\text{UE8M0}} = 2^{\lceil \log_2(\max_{j \in g} |y_{t,j}| / 448.0) \rceil}, \quad g \in \{0, \dots, 31\}$$
  其输出与 vLLM 的 `per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=fp8_sf_ue8m0)` 逐比特相等。
- **`sf_layout="nvfp4"` 契约**：在调用方提供的 `fp4_global_scale` 下，按每 16 列一组写出紧凑打包的 `y_fp4` (`uint8 [T, 2048]`，每字节 2 个 E2M1 元素) 与线性布局的 `y_fp4_sf` (`FP8-E4M3 [T, 256]`)，与 vLLM 的 `scaled_fp4_quant(y_bf16, fp4_global_scale, is_sf_swizzled_layout=False)` 完全一致。

### 2.2 Frost Low-Latency MoE 单一 TVM-FFI Host 入口 (`c0e51ae29b`)

在 `flashinfer/fused_moe/backends/frost_low_latency/pipeline.py` 与 `runtime.py` 中：
- **重构前**：每次 Warm 调用在 Python 侧对每个张量执行 `make_ptr(..., tensor.data_ptr(), ...)` 和 `from_dlpack(v)`，并分别调用 4 个编译好的 CuTe 函数（`pack_scale`、`pipeline.gate_up`、`pipeline.down_project`、`combine`）。
- **重构后**：在 `Pipeline` 上新增单一 `@cute.jit def launch(...)` 顶层方法，使用 `cute.compile(..., options="--enable-tvm-ffi")` 为每个 Tactic 编译 **唯一一个 TVM-FFI Host Entry**（缓存键前缀 `"mx-tc-ffi-v1"`）。指针重解释（`cute.recast_ptr`）与 Tensor 视图构造全部下沉到编译后的 FFI 函数内部执行。
- **安全性护栏**：`_COMPILED` 缓存仅保存不可变的可执行代码与几何特化元数据（`geometry, tile_k, pdl, wd, narrow_weights, finalize`），**绝不缓存可变张量指针、Workspace 地址或 CUDA Stream 状态**，从而在消除 75%+ Host 开销的同时保持多流与动态地址安全。

| Tactic (`T=1, H=2048, I=1408, E=64, top_k=6`, B300) | Public API 优化前 → 后 (µs) | Public 加速比 | Prepacked Runner 优化前 → 后 (µs) | Prepacked 加速比 | GPU Kernel 耗时 (µs) |
|---|---:|---:|---:|---:|---:|
| `BF16 FMA` | `106.04 → 35.81` | **2.96×** | `83.25 → 14.64` | **5.69×** | `26.63 → 26.65` |
| `MXFP4 TC128` | `260.85 → 61.85` | **4.22×** | `217.82 → 24.90` | **8.75×** | `10.33 → 10.33` |
| `MXFP4 FMA` | `154.30 → 47.30` | **3.26×** | `117.98 → 14.33` | **8.23×** | `17.25 → 17.25` |
| `MXFP8 TC128` | `256.21 → 60.98` | **4.20×** | `213.17 → 25.74` | **8.28×** | `10.50 → 10.50` |
| `MXFP8 FMA` | `152.79 → 47.38` | **3.22×** | `116.25 → 15.13` | **7.68×** | `24.52 → 24.52` |
| `NVFP4 FMA` | `162.43 → 48.37` | **3.36×** | `125.88 → 16.02` | **7.86×** | `13.23 → 13.23` |

### 2.3 TileLang 传递流水线依赖与跨线程域冲突分析 (`4b2032a6ce` & `1bafce1a37`)

1. **地址表达式依赖与传递闭包 (`pipeline_planning.cc`)**：`BufferRegionCollector::VisitExpr_(const BufferLoadNode *op)` 在遍历 `op->indices` 时临时保存并恢复 `is_global_read_`，从而将间接寻址 `A[idx_smem[i]]` 中的 `idx_smem` 准确记录为读取依赖而非被拷贝数据源；并通过单一反向工作表（Backward Worklist）求取标量与 Buffer 拷贝的传递生产者闭包，保留所有依赖拷贝的 `wait`。
2. **线程域独立展开 (`CollectPipelineThreadDomains` & `PipelineRegionsMayConflict`)**：在比较 `lhs` 与 `rhs` 两个 `BufferRegion` 之前，先分别将各自索引表达式中的线程绑定变量（`threadIdx.x` 等）在其完整定义域 $\mathcal{D}(v)$ 上求并集展开为 `IntSet`，再做区间求交：
   $$\text{Conflict}(R_1, R_2) \iff \forall d,\ \text{Eval}(R_{1,d}, \mathcal{D}_{\text{thread}}) \cap \text{Eval}(R_{2,d}, \mathcal{D}_{\text{thread}}) \neq \emptyset$$
   这样既能保留不同线程访问互不重叠 Shared Memory 切片时的流水线并行度，又能准确捕获跨线程访问同一物理区域的依赖。
3. **Ascend PTO MAD 方向谓词去重 (`1bafce1a37`)**：在 `PTOMmadDirectionAnalyzer::Merge` 构造 `S1 || S2` 之前，先检查指针对象同一性 (`same_as`) 与结构相等性 (`StructuralEqual`)，将 $N$ 个连续可变分支的谓词复杂度从 $\mathcal{O}(2^N)$ 降至 $\mathcal{O}(1)$。

---

## 3. 确定性实验验证结果 (`experiments/flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline`)

运行 `python experiments/flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline/run.py --check` 通过全部 8 项不变量校验：

- **Mega-mHC Hidden-4096 ($T=128$)**:
  - `fp8_group128` 写出 `[128, 4096]` FP8 + `[128, 32]` FP32 UE8M0 缩放因子（总计 `540,672 B`），消除独立量化算子对 `y_bf16` 的 `1,048,576 B` (`1.0 MiB`) HBM 重复读取。
  - `nvfp4` 写出 `[128, 2048]` Packed E2M1 + `[128, 256]` FP8-E4M3 缩放因子（总计 `294,912 B`），相比 BF16 载荷压缩比达 **3.5556×**。
- **Frost TVM-FFI 单入口优化**:
  - 在 6 组 B300 T=1 SwiGLU 算例中，Host 入口数从 `4` 降至 `1`，Public API 最高提速 **4.22×** (`MXFP4 TC128`: `260.85 → 61.85 µs`)，Prepacked Runner 最高提速 **8.75×** (`217.82 → 24.90 µs`)。
- **Cake StepFun MXFP8 Tile-128**:
  - $T=1024$ 默认 Tile 从 `64` 切换为 `128`，在 GB300 上实现 `1.047×` (`limit=7`) / `1.046×` (`limit=16`) 提速，在 B200 上实现 `1.044×` / `1.071×` 提速。
- **TileLang 流水线与 PTO 分析器**:
  - 反向工作表成功捕获传递生产者 `[1, 0]`；跨线程域 `[0, 63]` 与 `[1, 64]` 交集 `[1, 63]` 正确识别冲突，而不相交切片 `[0, 31]` vs `[32, 63]` 保持无冲突流水化；32 层可变守护的 PTO 谓词节点数从 `4,294,967,296` 降至 `1`。

---

## 4. 面向解决方案架构师的设计建议与可证伪问题

### 架构师检查清单
1. **何时启用 `sf_layout="fp8_group128"` vs `"nvfp4"`？** 当 `sm_107a` 模型的隐藏层维度为 `4096` 且紧接 DeepGEMM FP8 算子时，优先启用 `"fp8_group128"` (`fp8_sf_ue8m0=True`)；若紧接 Blackwell NVFP4 线性层，则传入预计算的 `fp4_global_scale` 并启用 `"nvfp4"`，避免任何独立 Quantize Kernel Launch。
2. **Decode 小 Batch ($T \le 4$) 的 Host Bottleneck 诊断**：在 B200/B300 上部署低延迟 MoE 时，若未使用全图 CUDA Graph，务必确认 CuTe-DSL 算子启用了单入口 `--enable-tvm-ffi` 打包，否则 CPU 端 DLPack 转换开销将是 GPU 执行时间的 8–20 倍。

### 后续可证伪实验问题
- **Q1**: 若在 `mega_mhc` Hidden-4096 NVFP4 变体中将 `is_sf_swizzled_layout` 从线性布局改为 128×4 Swizzled 布局，在 $T \le 64$ 的小 Token 场景下 Padding 开销是否会抵消下游 GEMM TMA 加载收益？
