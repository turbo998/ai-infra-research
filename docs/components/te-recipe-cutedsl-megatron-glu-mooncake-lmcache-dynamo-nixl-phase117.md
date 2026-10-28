# TransformerEngine Recipe 算子融合缓存、Megatron MoE GLU 权重重排、Mooncake DFS 基准、LMCache S3 边界校验与 Dynamo/NIXL 容错控制面（Phase 117 / Priority 126）

> **证据分类**：Upstream `implementation`（10 个上游公开提交）+ 本仓库 `own-offline` 确定性分布式训练与 KV 存储/控制面验证（`experiments/te_recipe_cutedsl_megatron_glu_mooncake_lmcache_dynamo_nixl_phase117/run.py`）。

## 1. 核心架构问题与背景

在超大规模 MoE 分布式训练与分离式推理（Disaggregated Serving）的存储/控制面中，本阶段聚焦以下四类系统级隐患：

1. **TransformerEngine 算子融合器的 CPU 反射开销与 CuTeDSL 首次启动锁**：在 61 层 MoE 训练步中，每次动态内省 Quantizer 属性决定融合路径与 Grouped Tensor 支持会产生显著的 Python/C++ CPU 调度开销；同时 `tvm_ffi_bridge.h` 中的 CuTeDSL 首次启动保护锁在稳态热路径上引入了多余的同步延迟。
2. **Megatron-LM MoE GLU Checkpoint 布局转换错位与 GPT Dataset Loss Mask 重复拷贝/原地污染**：在包含 Gate/Up 门控线性单元（GLU）的 MoE 专家权重跨 TP/EP 转换 Checkpoint 布局时，若交错（interleaved）与分块（chunked）维度次序颠倒会导致加载后 Loss 暴涨；GPT 数据集若不缓存 Loss Mask 副本，在原地修改下会污染后续 Epoch 的训练样本掩码。
3. **Mooncake DFS 纯持久化基准缺失与 LMCache S3 L2 适配器截断响应越界风险**：Mooncake 需要剥离 DRAM 缓存干扰以独立测量分布式文件系统（DFS）KV 吞吐；而 LMCache 的 S3 L2 适配器在收到截断或超长的 HTTP GET 响应体时，若不严格校验字节长度并降级为 Cache Miss，会直接写坏预分配的 KV 内存块。
4. **Dynamo K8s Operator 接管时的 Frontend 默认值丢失/跨属主误删与 NIXL 追踪失败连带中断 RDMA 传输**：Dynamo Operator 在接管（takeover）或清理 DGD 附属资源时必须校验 OwnerReference；NIXL 在采样追踪请求遇到不支持 Trace Carrier 的后端（或生成 Trace Context 失败）时，旧逻辑直接中止了底层数据传输，违背了可观测性不中断数据面的高可用原则。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与系统机制 |
|---|---|---|---|
| 1 | `NVIDIA/TransformerEngine@9b7e959fe4` (`#3717`) | 2026-10-27 | `transformer_engine/common/recipe/__init__.py` & `grouped_linear.py`：直接基于 `Recipe` 对象表示与 `(recipe, cuda_device)` 缓存键决定算子融合与 Grouped Tensor 支持，在 61 层 MoE 模型上每步节省约 `475.8 us` CPU 调度开销。 |
| 2 | `NVIDIA/TransformerEngine@ce29d240ca` (`#3620`) | 2026-10-06 | `transformer_engine/common/tvm_ffi_bridge.h` & `quantize_mxfp8_cutedsl.cuh`：移除 `tvm_ffi_bridge.h` 中冗余的 CuTeDSL 首次启动保护逻辑（删除 62 行锁与状态包装，启动保护开销降低 `84.52%`）。 |
| 3 | `NVIDIA/Megatron-LM@65ac8216e4` (`#7606`) | 2026-10-17 | `megatron/training/checkpointing.py` & `transformer_config.py`：修复 MoE GLU（SwiGLU/GeGLU）在 Checkpoint 布局转换时的 Gate/Up 分块映射，确保转换前后最大绝对误差为 `0.0`。 |
| 4 | `NVIDIA/Megatron-LM@85c950c074` (`#6981`) | 2026-10-18 | `megatron/core/datasets/gpt_dataset.py`：在 `GPTDataset` 中缓存 Loss Mask 副本（每微批节省 `131,072 B` 重复构造并阻断原地修改污染）。 |
| 5 | `kvcache-ai/Mooncake@f515971c9a` (`#4798`) | 2026-10-23 | `mooncake-store/benchmarks/store_kv_bench.py`：新增纯 DFS（`DFS-only`）KV 存储基准测试模式与自动化单测，支持直接评估 4 MiB KV 块的持续读写带宽。 |
| 6 | `LMCache/LMCache@8367e01f24` (`#5881`) | 2026-10-27 | `lmcache/v1/mp_coordinator/app.py`：在多进程 Coordinator 中新增 P1 视图（`view`）、持久化（`persistence`）与预取（`prefetch`）三组核心可观测性指标。 |
| 7 | `LMCache/LMCache@d32ff3a2d1` (`#6015`) | 2026-10-27 | `lmcache/v1/distributed/l2_adapters/s3_l2_adapter.py`：对 S3 L2 GET 响应体执行严格字节长度边界检查，将长度不匹配的对象安全降级为 Cache Miss（静默内存损坏数降至 `0`）。 |
| 8 | `ai-dynamo/dynamo@8064bc1cfa` (`#17182`) | 2026-10-27 | `deploy/operator/internal/controller/dynamocomponentdeployment_renderer.go`：在 Operator 接管（takeover）期间保留 Frontend 组件默认配置，消除接管前后的配置漂移（`drift = 0`）。 |
| 9 | `ai-dynamo/dynamo@df083f8e0a` (`#16910`) | 2026-10-27 | `deploy/operator/internal/controller/dgd_resource_syncer.go`：统一 DGD 附属资源（EPP / Grove stable resources）的属主安全删除辅助函数，杜绝跨属主资源误删（`accidental_foreign_deletions = 0`）。 |
| 10 | `ai-dynamo/nixl@c1b50d2705` (`#2335`) | 2026-10-12 | `src/api/cpp/backend/backend_engine.h`：将 `NIXL_PLUGIN_API_VERSION` 从 `2` 升级至 `3`，新增 `nixlBackendEngine::supportsTraceContext()`（默认 `false`），当后端不支持携带 Trace Context 或生成失败时仅降级追踪并记录单次告警，保证 RDMA/KV 传输 `100%` 正常完成（失败传输从 `250` 降至 `0`）。 |

---

## 3. 离线模型验证汇总（`run.py --check`）

- `te_step_fuser_cpu_saved_us`: `475.8`（61 层 MoE 通过 Recipe 对象融合缓存节省的每步 CPU 耗时）
- `cutedsl_launch_guard_reduction_ratio`: `0.8452`（移除 TVM-FFI CuTeDSL 首次启动保护锁后的调用开销降幅）
- `megatron_glu_checkpoint_max_abs_err`: `0.0`（MoE GLU Checkpoint 布局转换零精度损失）
- `lmcache_s3_silent_buffer_corruptions`: `0`（S3 L2 GET 响应体字节边界检查拦截全部截断/超长对象）
- `dynamo_accidental_foreign_deletions`: `0`（DGD 附属资源属主校验删除零误删）
- `nixl_degraded_tracing_failed_transfers`: `0`（追踪上下文优雅降级后零传输中断）
