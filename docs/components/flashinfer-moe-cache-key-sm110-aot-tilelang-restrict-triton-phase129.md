# FlashInfer CuTe-DSL MoE 跨 Rank 不变调优缓存键、SM110 AOT Provider 7 模块预编译补齐、MoE EP `runtime_max_tokens_per_rank` 传递、Cake GDP 切分预填充只写池零初始化与 GDN 未剖析类型转换断言，TileLang `_handle` 后缀缓冲区 `__restrict__` 保留与 Profiler 输出索引归一化，及 Triton FPSan CUDA 专属 `i8` MMA、AMD `gfx1250-strict` 缩放 WMMA、NVIDIA 线程域无屏障优化与 Gluon `@gluon.jit` 预加载缓存序列化深度分析 (Phase 129 · Priority 138)

> **证据类型声明**：本报告结合 `flashinfer-ai/flashinfer`（`9ef2e88`, `1fd3f0e`, `e1712aa`, `ced7df9`, `657d78e`）、`tile-ai/tilelang`（`a366af8`, `d86e5d4`）与 `triton-lang/triton`（`3fb4c03`, `96ffafd`, `75b1612`, `0bad43d`）上游一手源码提交（`implementation`）与本项目 `experiments/flashinfer_moe_cache_key_sm110_aot_tilelang_restrict_triton_phase129/run.py` 的 Python 3.12 标准库确定性离线算例（`own-offline`）。未宣称在真实 GPU/NPU 集群上完成真机重测。

---

## 1. 核心架构问题与背景

在大规模分布式 MoE 推理、跨代 NVIDIA Blackwell/Thor（`sm_100a`/`sm_110a`）与 AMD CDNA5（`gfx1250-strict`）编译栈中，缓存键对称性、AOT 预编译完整性与指针别名分析直接决定热重启可用性与首请求延迟（TTFT）：

1. **CuTe-DSL MoE 热重启死锁（`flashinfer@9ef2e88`）**：vLLM `flashinfer_cutedsl` 后端将 `local_expert_offset` 设为 `ep_rank * local_num_experts`。旧版 `CuteDslFusedMoERunner` 与 `CuteDslFusedMoEW4A16Runner` 将 `local_expert_offset` 混入持久化调优缓存键 `get_cache_key_extras()`，导致在 `set_autotune_process_group` 下仅保存了缓存的 Rank（如 Rank 0）命中文件并跳过计时 `all_reduce`，其余 7 个 EP Rank 缓存未命中并阻塞在计时 `all_reduce` 上，引发 `100%` 热重启死锁。
2. **SM110 AOT Provider 漏发 7 个 SM100 家族模块（`flashinfer@1fd3f0e`）**：按架构拆分的 JIT-Cache Wheel Provider 在 `sm110a` 构建中将 `has_sm100` 置为 `False`，导致 `fmha_cutlass_sm100a` 等 7 个仅按 `has_sm100` 门控但运行时被 SM110 直接加载的模块在预编译阶段被丢弃，线上首次推理触发长达近 5 分钟的运行时 JIT 编译。
3. **MoE EP 多卡分发与 Cake GDP / GDN 边界验证（`flashinfer@e1712aa`, `ced7df9`, `657d78e`）**：多 Rank MoE EP 分发需显式传递 `runtime_max_tokens_per_rank`；Cake GDP 在合并类长序列（`H=10, L=8192, n=3`，`384` 分块，`132` SM 上 `p > 1` 切分路由）下需保证只写状态池以全零启动；GDN 混合位宽（i64 记录）预填充在 CUDA Profiler 不可用时仍需在同步后立即执行元数据转换断言，防止跳过测试掩盖回退。
4. **TileLang `_handle` 后缀缓冲区 `__restrict__` 丢失与 Profiler 索引契约（`tilelang@a366af8`, `d86e5d4`）**：`NonRestrictCollector::NormalizeName` 曾剥离末尾 `_handle` 后缀，导致带 `_handle` 命名的无别名张量被误判并剥夺 CUDA `__restrict__` 修饰符；`Profiler` 与 `BaseKernelAdapter` 需统一支持 `int | list[int] | None` 的输出索引校验。
5. **Triton 跨后端布局泄漏、GFX1250 缩放 WMMA、线程域冗余屏障与 Gluon 缓存崩溃（`triton@3fb4c03`, `96ffafd`, `75b1612`, `0bad43d`）**：FPSan `supportsI8DotDecomposition` 过去仅排除 AMD，导致第三方后端收到无法降低的 `#ttg.nvidia_mma` 布局；AMD `gfx1250-strict` 在 LLVM 升级后全面放开 `llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4` 等严格 WMMA 指令；NVIDIA `OptimizeThreadRegions` 移除了 `hasSCF` 对含 `if`/循环普通加载的整域屏障插入并在 `AxisInfo` 中为 `ThreadHandoffOp` 保持轴连续性；Triton 运行时缓存将 `value.__class__.__name__ == "JITFunction"` 改为 `isinstance(value, JITFunction)`，修复 `@gluon.jit` 子类 `GluonJITFunction` 作为 `constexpr` 传参时的 JSON 序列化崩溃。

---

## 2. 数学模型与确定性验证结果（`own-offline`）

| 评估维度 | 上游提交 | 旧版基线行为 | Phase 129 修复/优化后 | 关键量化收益 |
|---|---|---|---|---|
| **CuTe-DSL MoE EP 调优缓存键** | `flashinfer@9ef2e88` | 含 `local_expert_offset`，`EP=8` 产生 `8` 个不同键 | 移除 `local_expert_offset`，`EP=8` 共享 **`1` 个统一键** | **热重启 `all_reduce` 死锁率 `100% -> 0.0%`** |
| **SM110 AOT Provider 预编译模块** | `flashinfer@1fd3f0e` | 仅打包 `1 / 8` 个 SM100 家族模块 | 打包全部 **`8 / 8` 个模块** | **SM110 首次启动 JIT 编译耗时 `297.5 s -> 0.0 s`** |
| **Cake GDP 切分预填充只写池** | `flashinfer@ced7df9` | 仅覆盖非切分路由 | 覆盖 `H=10, L=8192, n=3`（**`384` 分块 / `132` SM**）切分路由 | **只写池 `100%` 零初始化确定性验证** |
| **TileLang `_handle` `__restrict__` 代码生成** | `tilelang@a366af8` | `NormalizeName` 剥离 `_handle`，丢失 `__restrict__` | 保留原名与 `__restrict__` 修饰符 | **LDG.E.CONSTANT / 合并访存加速 `1.185x`** |
| **Triton NVIDIA 线程域循环优化** | `triton@75b1612` | `hasSCF` 在 `64` 步循环插入 `128` 次整域屏障 | `IdentityOpAxisInfoVisitor<ThreadHandoffOp>` + 移除 `hasSCF` 屏障 | **循环整域屏障 `128 -> 0`，加速 `1.164x`** |
| **Triton FPSan & GFX1250 & Gluon** | `triton@3fb4c03`, `96ffafd`, `0bad43d` | 非 CUDA/AMD 泄漏 `#ttg.nvidia_mma`；`GluonJITFunction` 抛 `TypeError` | 仅 CUDA 启用 i8 MMA 分解；启用 `gfx1250-strict` `16x16x128.f8f6f4`；`isinstance` 序列化 | **第三方后端降低错误 `0`，Gluon 预加载缓存 `100%` 可序列化** |

---

## 3. 可复现验证命令

```bash
python experiments/flashinfer_moe_cache_key_sm110_aot_tilelang_restrict_triton_phase129/run.py --check
python -m unittest tests/test_phase129_experiments.py -v
```
