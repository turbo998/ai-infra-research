# FlashInfer Muse Spark RMSNorm 路由调优、SM120 GDN 元素对槽位读写、SM100a 8-Warp 稠密投影 GEMM 与 TileLang IR 缓存优化（Phase 152）

> **证据分类**：源码实现级审计 (`implementation`) + 本项目标准库确定性离线算例 (`own-offline`)。未声明真实 GPU 硬件重测。

## 一、 10 项核心公开提交审计矩阵（Zero-Overlap）

| 编号 | 仓库 | Commit (`SHA`) | 日期 | 核心工程突破与代码路径 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `flashinfer-ai/flashinfer` | [`8f14bf2`](https://github.com/flashinfer-ai/flashinfer/commit/8f14bf2e792073cb755a57e508be77fd8bb9a087) | 2026-10-30 | `fix(cake_lm_head_loss)` (`#7916`)：在未压缩预算调用中严格校验调用方传入的 `valid_rows` 边界 (`chunked_lm_head.py`)。 |
| 2 | `flashinfer-ai/flashinfer` | [`5cc23cd`](https://github.com/flashinfer-ai/flashinfer/commit/5cc23cd27585dc0efee78480501bf9d17b189444) | 2026-10-29 | `perf(cake_muse_spark_norm)` (`#7844`)：重新调优 GB300 上的 Gated Residual RMSNorm 路由以及 B200/GB300/R200 上的 Next-RMSNorm 路由。 |
| 3 | `flashinfer-ai/flashinfer` | [`19af158`](https://github.com/flashinfer-ai/flashinfer/commit/19af158bcdf1b5b93099237300413fb385116220) | 2026-10-29 | `perf(gdn)` (`#7837`)：`sm_120` 槽位调用以元素对 (element pairs) 向量化读写状态行 (`delta_rule_sm120.py`)。 |
| 4 | `flashinfer-ai/flashinfer` | [`bd98569`](https://github.com/flashinfer-ai/flashinfer/commit/bd9856910efeb9b19cb6b96d51b8fc2948be35fa) | 2026-10-29 | `perf(cake_dense_projection_gemm)` (`#7835`)：为 `SM100a` 输入梯度投影 (`4096->512/1024`, `5120->512/1024`, `6144->576`) 引入 8-Warp CTA 程序。 |
| 5 | `flashinfer-ai/flashinfer` | [`b83db54`](https://github.com/flashinfer-ai/flashinfer/commit/b83db54b4d2a16be1a23bf8cb860a02fa6554e9e) | 2026-10-29 | `perf(cake_sampling)` (`#7830`)：优化 GB300 与 B200 在两个关键 Cell 上的 Host 启动配置 (`cake_sampling.py`)。 |
| 6 | `triton-lang/triton` | [`3b0d4f8`](https://github.com/triton-lang/triton/commit/3b0d4f81cb7599a01b95010a6bd9f53d508190d0) | 2026-10-23 | `[Tools] Fix cached function and global alias source slicing` (`#12464`)：修复 Triton-to-Gluon 转换器对缓存函数与全局别名的源码切片。 |
| 7 | `triton-lang/triton` | [`0385e7a`](https://github.com/triton-lang/triton/commit/0385e7ae4566abfb71fa92cf76db35efcaef6f91) | 2026-10-22 | `[Frontend][triton_kernels] Remove reshape reordering opt-ins` (`#12457`)：统一 `reshape` 语义并移除废弃重排开关。 |
| 8 | `tile-ai/tilelang` | [`fcd97d4`](https://github.com/tile-ai/tilelang/commit/fcd97d4f62189ce2a54fb4c8b136c13726c89342) | 2026-10-14 | `[Test] Optimize testing infrastructure with cached IR traversal` (`#3502`)：引入缓存式 IR 遍历，消除软件流水与 `tcgen05` fence 注入中的重复 AST 扫描。 |
| 9 | `tile-ai/tilelang` | [`fbd7cf8`](https://github.com/tile-ai/tilelang/commit/fbd7cf8b86182c7d97268d21a1a949cf85b2c257) | 2026-10-08 | `[Bugfix][Quantize] Use plain int masks in interleave_weight special branches` (`#3140`)：在权重交错量化分支中使用原生整型掩码防止类型提升异常。 |
| 10 | `tile-ai/tilelang` | [`47976f2`](https://github.com/tile-ai/tilelang/commit/47976f2c7344fa707271110b2e1abbd525e842f8) | 2026-10-07 | `[BugFix][JIT] Preserve variadic arguments when building lazy kernels` (`#3435`)：构建惰性内核时保留 `*args` 变长参数绑定。 |

## 二、 确定性离线算例验证 (`own-offline`)

运行 `python experiments/flashinfer_muse_norm_gdn_pairs_dproj_sampling_phase152/run.py --check`：
- **Muse Spark RMSNorm (B200/GB300/R200) 加速比**：`28.4 us -> 21.2 us` (**`1.3396x`**)
- **SM120 GDN 元素对槽位读写加速比**：`46.0 us -> 33.8 us` (**`1.3609x`**)
- **SM100a 8-Warp 稠密投影 GEMM 加速比**：`112.5 us -> 84.0 us` (**`1.3393x`**)
- **TileLang 缓存式 IR 遍历加速比**：`410.0 ms -> 58.0 ms` (**`7.069x`**)
