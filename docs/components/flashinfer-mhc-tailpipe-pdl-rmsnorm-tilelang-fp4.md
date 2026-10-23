# FlashInfer SM107a MHC Tailpipe 流水折叠、SM90 PDL RMSNorm RAW/WAR 排序契约、SM110a Thor 二轮去冗余与 TileLang Ascend FP4/WAW/MMA 护栏深度解析

> **语言 / Language**: [中文](flashinfer-mhc-tailpipe-pdl-rmsnorm-tilelang-fp4.md) | [English](flashinfer-mhc-tailpipe-pdl-rmsnorm-tilelang-fp4.en.md)  
> **Phase**: 68 (`priority: 77`, Track A — GPU Kernel & Quantization)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `author-benchmark` (作者 R200/GH200/Thor 实测报告) + `own-offline` (本项目确定性标准库仿真 `experiments/flashinfer-mhc-tailpipe-pdl-rmsnorm-tilelang-fp4/run.py`)

---

## 1. 架构背景与核心工程问题

在 Blackwell/Rubin (`sm_107a`)、Hopper (`sm_90`)、Jetson AGX Thor (`sm_110a`) 与华为昇腾 Ascend NPU 的异构算子工程中，算子调度器与编译器必须在**流水线重叠收益**、**跨网格数据依赖安全（PDL）**、**大规模生成代码的可维护性**以及**亚字节（FP4）块缩放对齐/共享内存屏障推导**之间建立严格的数学与编译契约：

1. **FlashInfer `sm_107a` Hidden-4096 Mega-MHC Tailpipe 流水线切换 (`flashinfer-ai/flashinfer@6103f5ea7c89`, `#7121`)**：当 Multi-Head Latent/Compression (MHC) 尾端归约算子在 `hidden=4096`、BF16 输出下运行时，让尾端（tail）第 1 轮加载与前序折叠（fold）重叠（即 `"tailpipe"` 体制）是否总是更优？为什么在 `splits=32/64` 的高切分短序列区反而需要保留 `"bf16"` 体制？
2. **FlashInfer `sm_90` PDL (`enable_pdl=True`) 下 `rmsnorm` / `fused_add_rmsnorm` 的 RAW/WAR 排序契约 (`flashinfer-ai/flashinfer@d51e1660b22c`, `#6799`)**：Programmatic Dependent Launch (PDL) 允许后继 kernel 在前驱 kernel 尚未完全退出时提前启动 preamble。如果从 2D `rmsnorm` kernel 中移除 `cudaGridDependencySynchronize()` (`griddepcontrol.wait`)，在原地（in-place）读写与跨流重叠场景下会发生什么程度的数值破坏？
3. **FlashInfer `sm_110a` Thor (`sm110_xqa` + `sm110_gqa_decode`) 第二轮去冗余重构 (`flashinfer-ai/flashinfer@9143f50102b0`, `#7116`)**：如何在消除 `50` 个编译单元中 `14,131` 行净冗余代码的同时，保证 `50/50` 单元的 SASS 机器码与预处理行号契约 `100%` 恒等？
4. **TileLang Ascend FP4×FP4 DeepGEMM 块缩放对齐、共享内存同迭代跨线程 WAW 屏障与 CUDA MMA K-atom 显式填充 (`tile-ai/tilelang@baceed6676df` `#3589`, `@eaa9468d60f2` `#2821`, `@876a59e6eb80` `#2818`)**：当操作数位宽从 FP8 (`8-bit`) 缩减至打包 FP4 (`float4_e2m1fn_x2`, `4-bit`) 时，MN-major 块缩放加载对齐粒度如何变化？编译器如何区分需要同步屏障的跨线程同迭代 WAW 与无需屏障的不相交分区？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 FlashInfer `sm_107a` Hidden-4096 Mega-MHC `"tailpipe"` vs `"bf16"` 双体制分派 (`#7121`)

在 `flashinfer/experimental/sm107_mega_mhc/api.py` 与 `csrc/cake_mega_mhc_c549b064530c3765f9ba_{kernel,binding}.cu` 中，FlashInfer 为 `sm_107a` 新增了第二个 Hidden-4096 BF16 输出算子 `cake_mega_mhc_c549b064530c3765f9ba`（`"tailpipe"` 体制），与原有的 `cake_mega_mhc_f24c2f3f98bfe3356398`（`"bf16"` 体制）并存：

- **数据移动与流水重叠**：在 `"tailpipe"` 体制中，tail 阶段的第 1 轮（round-1）全局内存加载与 fold 阶段重叠执行。当切分数 `splits <= 22` 时，每个 split 块承载的 token 行数充足，重叠加载能够隐藏 HBM 延迟；而在 `splits in {32, 64}`（对应 208-SM GPU 上 `T <= 256`）时，每个 split 块极小，重叠引发的同步与寄存器压力抵消收益。
- **分派公式 (`mhc_infer_tail_program` & `mhc_infer_tail_splits`)**：
  $$
  \text{regime}(T, \text{SM}) = \begin{cases}
  \texttt{"tailpipe"} \;(\texttt{c549b064530c3765f9ba}), & \text{if } \text{splits}(T, \text{SM}) \le 22 \\
  \texttt{"bf16"} \;(\texttt{f24c2f3f98bfe3356398}), & \text{if } \text{splits}(T, \text{SM}) > 22
  \end{cases}
  $$
  - 在 `208 / 212 / 216` SM 设备（如 NVIDIA R200 208 SMs）上：`T <= 128 -> 32 splits` (`"bf16"`)，`128 < T <= 256 -> 64 splits` (`"bf16"`)，`256 < T <= 512 -> 22 splits` (`"tailpipe"`)，`512 < T <= 768 -> 16 splits` (`"tailpipe"`)，`768 < T <= 1024 -> 22 splits` (`"tailpipe"`)，`1024 < T <= 2048 -> 13 splits` (`"tailpipe"`)，`T > 2048 -> 16 splits` (`"tailpipe"`)。切换边界精确位于 $T \ge 385$（在连续基准中 $T \ge 257$）。
  - 在 `148 / 152` SM 设备上：`T <= 128 -> 32 splits`，`128 < T <= 256 -> 64 splits`，`256 < T <= 384 -> 15 splits` (`"tailpipe"`)，因此 $T \ge 257$ 即切入 `"tailpipe"`。
- **作者实测性能 (`author-benchmark`, NVIDIA R200 `sm_107a` 208 SMs)**：两种体制输出逐位完全一致（bitwise-identical）。在 $T=512$ (`22` splits) 下延迟从 `14.016 us` 降至 `13.888 us` (`1.009x`)；$T=2048$ (`13` splits) 从 `28.408 us` 降至 `27.944 us` (`1.017x`)；$T=4096$ (`16` splits) 从 `55.512 us` 降至 `54.208 us` (`1.024x`)；$T=8192$ (`16` splits) 从 `112.656 us` 降至 `111.103 us` (`1.014x`)。

### 2.2 FlashInfer `sm_90` PDL `rmsnorm` / `fused_add_rmsnorm` RAW/WAR 排序契约 (`#6799`)

在 `tests/norm/test_rmsnorm_pdl.py` 与 `csrc/norm.cu` 中，FlashInfer 构建了针对 `enable_pdl=True` 的完整 RAW（Read-After-Write）与 WAR（Write-After-Read）乱序竞争验证套件：

- **测试矩阵与自定义流并发探针**：覆盖 5 个 2D 形状（`264×128` 单ブロック单轮、`264×1024` 单轮向量化、`264×4096` 多轮、`264×8192` 大型多轮、`264×16384` CTA 间归约）与 3 个 3D QK-norm 形状（`64×8×128`、`33×8×256`、`17×4×512`），跨越 `float16`/`bfloat16` 与 eager/`cuda.make_graphed_callables`，总计 **242 个测试用例**（160 个输出契约用例 + 82 个 PDL 竞争用例）。
- **两类 PDL 危险窗口**：
  1. `fill -> norm (RAW)`：前驱生产者向全局内存写入激活值，后继 `rmsnorm(enable_pdl=True)` 必须在发射全局内存读取前执行 `griddepcontrol.wait`。
  2. `delayedproducer -> in-place norm (WAR + RAW)`：当 `rmsnorm(x, weight, out=x, enable_pdl=True)` 原地覆盖输入，或 `fused_add_rmsnorm(x, residual, weight, enable_pdl=True)` 同时原地修改 `x` 与 `residual` 时，若缺乏 `griddepcontrol.wait`，后继 kernel 会在前驱仍在读取/写入时提前覆写同一缓存行。
- **反证实验 (`author-benchmark`, GH200 `sm_90` 132 SMs)**：当保留 `griddepcontrol.wait` 时，`242/242` 全部通过；当从 `csrc/norm/norm.cuh` 的 4 个 2D kernel 中注释掉 `cudaGridDependencySynchronize()` 时，66 个竞争测试中有 **56 个 2D 案例立即失败**（另外 10 个未改动的 3D QK 与探针正对照通过），例如 `rmsnorm-264x128 bf16 eager` 的 `33,792 / 33,792` (`100%`) 元素全部损坏，`264x16384` 的 `3,899,392 / 4,325,376` (`90.15%`) 元素损坏。此外，将 `csrc/norm.cu` 的 `#include <ATen/cuda/CUDAContext.h>` 替换为 `<c10/cuda/CUDAStream.h>` 消除了 `cuComplex.h`、`cublas_v2.h`、`cusparse.h`、`cusolverDn.h` 等 6 个重型头文件依赖。

### 2.3 FlashInfer `sm_110a` Thor (`sm110_xqa` + `sm110_gqa_decode`) 第二轮零差异去冗余 (`#7116`)

针对 Jetson AGX Thor (`sm_110a`) 的 43 个 XQA 算子与 7 个 GQA Decode 算子（共 50 个 kernel + 50 个 binding），`#7116` 将重复样板折叠至 9 个共享头文件，净删减 **14,131 行**（`+1,729 / -15,860`）：
- **Device ABI 与按需宏门控**：`cake_sm110_xqa_device_abi.cuh` 与 `cake_sm110_gqa_decode_device_abi.cuh` 统一 LP64 `uint64_t`、`CakeTensorMap64`、`CUtensorMap` 与 `make_warp_uniform`；`cake_sm110_xqa_device_helpers.cuh`（32 个内联 helper）与 `cake_sm110_gqa_decode_device_helpers.cuh`（19 个内联 helper）通过 `428` 处 `#define CAKE_<PKG>_NEED_*` 按需展开，并在头文件尾部 `#undef` 全部选择器，确保预处理文本与重构前逐字符匹配。
- **`KVCacheList` 与 `__LINE__` 偏移契约**：`cake_sm110_xqa_kv_cache_list.h` 按 `#if USE_PAGED_KV` 提供 `contig`（3 字段）与 `page128`（4 字段，含 `k_page_table`/`v_page_table`）定义，覆盖 80 个编译单元。49 个 TMA/non-TMA binding 中共迁移 `248` 处 `TVM_FFI_CHECK` 调用点，通过精确记录行号偏移（XQA `51 / 48 / 34` 行，GQA `14 / 16` 行），在 Thor 上实现 `50/50` 单元 `cuobjdump --dump-sass` 完全一致。

### 2.4 TileLang Ascend FP4×FP4 块缩放对齐、同迭代跨线程 WAW 与 CUDA MMA K-atom 护栏 (`#3589`, `#2821`, `#2818`)

1. **Ascend Packed FP4 (`float4_e2m1fn_x2`) DeepGEMM (`#3589`)**：
   - 两个 FP4 E2M1 值打包进 1 个 `int8`/`float4_e2m1fn_x2` 字节，低半字节（lower nibble）优先沿连续轴排列。
   - 在 `examples/ascend/deepgemm/kernels/{compute,epilogue}.py` 中，当 `blockscaled=True` 且操作数为 `Major.MN` 时，单次 MAD tile 的有效加载高度/宽度对齐公式从硬编码 `32` 推广为：
     $$
     \text{load\_mad\_mn} = \operatorname{align\_up}\!\left(\text{eff\_mad\_mn},\; \frac{256}{\text{operand\_bits}}\right)
     $$
     对于 FP8 (`operand_bits = 8`)，对齐粒度为 $256 / 8 = 32$；对于 FP4 (`operand_bits = 4`)，对齐粒度翻倍为 $256 / 4 = 64$（对应 32 字节硬件突发对齐）。同时 `src/ascend/transform/insert_oob_padding.cc` 的 `MakeMxKTailFill` 支持自动为 `is_float4_e2m1fn()` 发射 L1 K-tail 零填充。
2. **共享内存同迭代跨线程 WAW 冲突检测 (`#2821`)**：
   - 在 `src/transform/thread_storage_sync.cc` 的 `TileLangThreadSyncPlanner` 中，此前 WAW 与 RAR 统一复用相同的线程符号变量（仅检查单线程跨迭代自覆盖）。但当 `loop == nullptr && prev.type == kWrite && curr.type == kWrite`（同迭代 WAW）时，不同线程可能写入同一共享内存地址（例如 `T.copy(A, a_shared)` 后接 `T.fill(a_shared[10:16], 99.0)`，其中 `tx < 6` 写 `s[tx + 10]` 与前序 `tx' in [10, 15]` 的写入重叠）。
   - 修复后令 `use_distinct_threads = (prev.type != curr.type) || is_same_iteration_waw`，通过 Presburger 约束求解器仅在跨线程写集合非空交集时插入 `__syncthreads()`，而对不相交分区（如 `s[tx]` 与 `s[tx + 64]`）或线程私有重复写保持零屏障。
3. **CUDA MMA K-atom 整除校验与显式零填充 (`#2818`)**：
   - 在 `tilelang/cuda/intrinsics/macro/mma_macro_generator.py` 中，通用 CUDA MMA emitter 的 K-atom 为 $K_{\text{atom}} = 256 / \text{bits}$（`sm_80` 下 FP16/BF16 为 `16`，INT8 为 `32`，INT4 为 `64`；`sm_75` 下为 $128 / \text{bits}$）。此前 `min(256 // bits, chunk)` 会静默截断不整除的 `chunk`，导致生成非法的 PTX `mma.sync` 索引。修复后当 `chunk % k_dim != 0` 时直接抛出 `ValueError`，提示用户将 K tile 显式 pad 到 $\lceil \text{chunk} / K_{\text{atom}} \rceil \cdot K_{\text{atom}}$ 并对 A、B 两侧补零。

---

## 3. 本项目确定性离线实验验证 (`own-offline`)

运行 `python experiments/flashinfer-mhc-tailpipe-pdl-rmsnorm-tilelang-fp4/run.py --check`，产出 `summary.json`，核心验证结果如下表：

| 验证维度 | 指标 / 场景 | 基线 / 负对照 | 优化 / 护栏后结果 | 证据等级 |
|---|---|---|---|---|
| **SM107a MHC Tailpipe 分派 (`#7121`)** | R200 (208 SMs) `T=256` vs `T=512` vs `T=4096` | `splits=64` (`13.920 us`) / `splits=22` (`14.016 us`) / `splits=16` (`55.512 us`) | `T<=256` 保持 `"bf16"`；`T=512` 选 `"tailpipe"` (`13.888 us`, `1.009x`)；`T=4096` 达 `54.208 us` (`1.024x`) | `author-benchmark` + `own-offline` |
| **SM90 PDL RMSNorm 排序契约 (`#6799`)** | GH200 (132 SMs) 242 用例 / 移除 2D `griddepcontrol.wait` | 移除 wait 后 `56/66` 竞争用例失败（`264x128` `100%` 元素损坏） | 保留 `griddepcontrol.wait` 下 `242/242` 全通过（`0` 损坏），头文件减少 `6` 个 | `author-benchmark` + `own-offline` |
| **SM110a Thor 去冗余 (`#7116`)** | 43 XQA + 7 GQA Decode 编译单元 | 15,860 行冗余 ABI/Helper/Check 代码 | 折叠进 `9` 个头文件，净减 `14,131` 行，`50/50` SASS 完全一致 | `implementation` + `own-offline` |
| **TileLang FP4 / WAW / MMA (`#3589/#2821/#2818`)** | `eff_mad_m=18` 对齐；`s[tx]` vs `s[tx+10]` WAW；`sm_80` FP16 `K=20` | FP8 对齐 `32`；同变量 WAW 漏检屏障；`min(16, 20)` 静默截断 | FP4 MN-major 对齐 `64`；偏移重叠自动插 `1` 个 barrier（不相交 `0` 个）；`K=20` 报错要求 pad 至 `32` | `implementation` + `own-offline` |

---

## 4. 面向解决方案架构师的练习与可证伪问题

1. **计算题（MHC 尾端分派与切分粒度）**：给定一台 `148` SM 的 GPU 与一台 `208` SM 的 R200 GPU，当输入序列长度 $T=300$、`hidden=4096`、`dtype=bfloat16` 时，`mhc_infer_tail_splits(300, sm_count)` 在两台设备上分别返回多少个 splits？各自会选择 `"tailpipe"` 还是 `"bf16"` kernel？
   - *参考解答*：在 `148` SM 上，$256 < 300 \le 384$，返回 `15` splits ($\le 22$)，选择 `"tailpipe"` (`cake_mega_mhc_c549b064530c3765f9ba`)；在 `208` SM 上，$256 < 300 \le 384$，返回 `22` splits ($\le 22$)，同样选择 `"tailpipe"`。若 $T=256$，两者均返回 `64` splits ($> 22$)，均退回 `"bf16"`。
2. **架构设计题（PDL 流水线中的 In-Place 算子护栏）**：在设计自定义 CUDA Graph 推理管线并开启 `enable_pdl=True` 时，为什么即使单算子静态输出测试（eager 单步调用）全部通过，线上并发推理仍可能出现间歇性 NaN？应如何在 CI 中构造确定性复现探针？
   - *参考解答*：默认流或空载 GPU 上前驱 kernel 往往在后继 kernel 发射全局读取前就已自然排空；必须像 `#6799` 一样在共享 SM 资源的辅助流上注入高占用 `spin_kernel`，并使用长尾延迟的 `delayedproducer` 主动拉开前驱尾部写入与后继 preamble 的重叠窗口，才能稳定捕获缺失 `griddepcontrol.wait` 的 RAW/WAR 竞态。
3. **后续可证伪问题**：在 `tile-ai/tilelang` 的 Ascend DeepGEMM 中，当前 `("float8_e4m3fn", "float4_e2m1fn_x2")` 混合精度（激活 FP8 × 权重 FP4）仍标记为 `NotImplementedError("FP8 x FP4 (FP8 dequant) kernels are not implemented")`。后续当 L1->L0 在线解量化或 AIV 解量化通路上线后，其 MN-major 块缩放对齐粒度应由激活（32）还是权重（64）主导？
