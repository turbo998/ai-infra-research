# FlashInfer Cake NVFP4 B200 `M=1` PDL 链与 GB300 折叠缩放多播 GEMM、Blackwell Softmax 路由、Causal Conv1D 双阶段反向、GDN 伴随磁带核、Frost MX 跨架构加载，及 Triton MMAv5 `warpid=0` 折叠与 CDNA3/4 缓冲原子操作分析（Phase 131 · Priority 140）

> **English Version**: [flashinfer-cake-fp4-conv-gdn-triton-warp-fold-cdna-phase131.en.md](flashinfer-cake-fp4-conv-gdn-triton-warp-fold-cdna-phase131.en.md)  
> **证据锚点**: `ev-flashinfer-cake-fp4-conv-gdn-triton-warp-fold-cdna-phase131-2026` (`own-offline` + `implementation`)  
> **可复现实验**: `python experiments/flashinfer_cake_fp4_conv_gdn_triton_warp_fold_cdna_phase131/run.py --check`

---

## 1. 第一手源码证据矩阵（Primary Source Matrix）

| 仓库与 Commit | 源码路径 | 核心架构变更 |
|---|---|---|
| `flashinfer-ai/flashinfer` [`449a357`](https://github.com/flashinfer-ai/flashinfer/commit/449a35734465609bd236fc52d6cac1e84ba2dde7) | `flashinfer/experimental/cake_nvfp4_per_token/cake_backend.py` | B200 (`sm_100a`) `M=1` Programmatic Dependent Launch (PDL) 融合 `quantize -> GEMM` 链，以及 GB300 (`sm_103a`) 面向 `M=8192, K=28672` 将 `out_scale` 折叠入逐 Token 激活量化的多播 NVFP4 GEMM |
| `flashinfer-ai/flashinfer` [`59f1b2c`](https://github.com/flashinfer-ai/flashinfer/commit/59f1b2c2a1aca08b2104e6a939dee7c149e39521) | `csrc/cake_blackwell_softmax.cu` | 更新 Blackwell (`sm_100a` / `sm_103a`) 20 个 Cake Softmax 程序的批窗口分派规则（Cluster 大小与 Chunk 映射） |
| `flashinfer-ai/flashinfer` [`500833a`](https://github.com/flashinfer-ai/flashinfer/commit/500833a11d3abf4b9a11ea9380777142731e4ec6) | `flashinfer/cake_causal_conv1d_train_backend/cake_jit.py` | `cake_causal_conv1d_train` 反向传播在 `C < 512` 引入双阶段（stage-pair）回退程序并在优势行窗口启用全 Rank 主网格 |
| `flashinfer-ai/flashinfer` [`ff078b4`](https://github.com/flashinfer-ai/flashinfer/commit/ff078b40208517d0aa14f891302297afbe17ebbd) | `flashinfer/experimental/cake_gdn_train/cake_jit.py` | 加速上下文并行 GDN 训练核（生产形状 `H_qk=16, HV=48`）的前向预处理与伴随遍历（adjoint-walk）磁带核 |
| `flashinfer-ai/flashinfer` [`bdb35ef`](https://github.com/flashinfer-ai/flashinfer/commit/bdb35ef12a8df6669e4de3d749f761aaa16b6f41) | `flashinfer/fused_moe/backends/frost_low_latency/runtime.py` | 在 CuTe-DSL 跨架构冷编译未附带 JIT 引擎时，显式通过目标导出与 TVM-FFI 模块加载器物化 Frost MX 可执行文件 |
| `triton-lang/triton` [`1b4a204`](https://github.com/triton-lang/triton/commit/1b4a2041b0e3eda43ad398636098ef9147e4663c) | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp` | 在 Warp Specialization 转 LLVM 阶段将单 Warp 分区的相对 `warpid` 常量折叠为 `0`，消除 MMAv5 `tcgen05.mma` 发射前的条件分支 |
| `triton-lang/triton` [`a950b39`](https://github.com/triton-lang/triton/commit/a950b397451d3e90c9ba6d9e29be3bf470e2ac0b) | `third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp` | 在 AMD 多 CTA（Multi-CTA Cluster）核中从前端与编译器层禁止不可达的标量原子操作 |
| `triton-lang/triton` [`71114a1`](https://github.com/triton-lang/triton/commit/71114a1107e6deb62d52fb5e6ac82530db6961b5) | `python/src/gluon_ir.cc` | 新增 `amdg.buffer_atomic_load` / `amdg.buffer_atomic_store`，降级至 `llvm.amdgcn.raw.ptr.atomic.buffer.load` 防止轮询标志被 LLVM 外提 |
| `triton-lang/triton` [`dc1d703`](https://github.com/triton-lang/triton/commit/dc1d703e847b599603095b2a04b41e7794a0c318) | `python/triton/experimental/gluon/language/amd/cdna3/__init__.py` | 将 `gfx9` `CachePolicy` 迁移至 `cdna3` / `cdna4` 架构专属命名空间 |
| `triton-lang/triton` [`cb0f110`](https://github.com/triton-lang/triton/commit/cb0f110def4df43728fcd8c450eb31251abd786f) | `third_party/nvidia/backend/driver.py` | 缓存 NVIDIA CUDA 设备属性查询，消除热路径重复驱动调用 |

---

## 2. 核心架构机制与确定性建模（Core Architectural Mechanisms）

1. **Cake NVFP4 `M=1` PDL 链与 GB300 多播折叠缩放（`449a357`）**：
   - 在 B200 解码单请求（`M=1`）场景下，通过 Programmatic Dependent Launch (PDL) 将逐 Token 激活量化与 NVFP4 GEMM 链接执行，延迟从 **`18.4 us` 降至 `12.6 us`（提速 `1.4603x`）**。
   - 在 GB300 大批量 Prefill（`M=8192, N=7168, K=28672`）下，将 GEMM 输出缩放因子 `out_scale` 折叠进激活量化器，消除额外 BF16 缩放读写通道（省去 **`234,881,024 B` / `224.0 MiB` HBM 流量**），结合 TMA 多播将核耗时从 **`412.0 us` 压降至 `338.5 us`（提速 `1.2171x`）**。
2. **Cake Causal Conv1D 双阶段反向（`500833a`）、GDN 伴随遍历磁带核（`ff078b4`）与 Frost MX 跨架构加载（`bdb35ef`）**：
   - `cake_causal_conv1d_train` 在 `C < 512` 的回退路径引入 stage-pair 反向核，延迟从 **`94.0 us` 降至 `71.5 us`（`1.3147x`）**；GDN 训练核（`H_qk=16, HV=48`，每 Q/K 头聚合 `3` 个 Value 头）优化前向准备与伴随磁带遍历，耗时从 **`168.0 us` 降至 `128.0 us`（`1.3125x`）**；Frost MX 在跨架构编译时显式物化 TVM-FFI 模块，将 `TVM FFI function is not initialized` 异常率从 **`100%` 归零**。
3. **Triton MMAv5 单 Warp `warpid=0` 折叠（`1b4a204`）与 AMD CDNA3/4 缓冲原子操作（`71114a1`）**：
   - 当 Warp Specialization 分区仅分配 `1` 个 Warp 时，`ConvertWarpSpecializeToLLVM` 将相对 `warpid` 常量折叠为 `0`，使 LLVM 消除 `K=128` 步循环内 **`128 -> 0` 次 `tcgen05.mma` 发射前分支判断**；同时 `cdna{3,4}.buffer.atomic_load` 降级至不可被 LLVM LICM 外提的 `llvm.amdgcn.raw.ptr.atomic.buffer.load`，彻底消除设备端自旋轮询死锁。
