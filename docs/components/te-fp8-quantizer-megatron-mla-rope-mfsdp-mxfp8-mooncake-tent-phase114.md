# TransformerEngine 实例级 FP8 量化器参数、Megatron 连续 MLA RoPE 访存与 MFSDP v2 MXFP8、TorchTitan MoE/RL、Mooncake MemFabric/DFS、LMCache 与 NIXL HF3FS（Phase 114）

> **证据分类**: `implementation` + `own-offline`（上游主干源码分析与可复现离线算例验证，非真实 GPU/RDMA 硬件 profiling）

## 1. 核心源码改动与分布式系统机制

| Evidence ID | 上游提交 | 关键实现路径与分布式系统不变量 |
|---|---|---|
| `te-build-fp8-recipe-quantizer-params-per-instance-2026` | `NVIDIA/TransformerEngine@14be43fb92` (`2026-10-08`) | `transformer_engine/common/recipe/__init__.py` 在 `Float8CurrentScaling` 每次实例化时动态构建独立的 `QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)`，修复类属性共享导致的跨实例状态污染与 `use_power_2_scales` 失效。 |
| `te-common-enable-all-activations-mxfp8-cpp-test-2026` | `NVIDIA/TransformerEngine@f57b8b8e61` (`2026-10-09`) | `tests/cpp/operator/test_cast_mxfp8*.cu` 为单张量与 grouped MXFP8 cast C++ 算子开启全部 5 类激活（`GeLU`、`SiLU`、`ReLU`、`QGeLU`、`SReLU`）。 |
| `megatron-mla-query-rope-contiguous-forward-load-2026` | `NVIDIA/Megatron-LM@a4b2beebc0` (`2026-10-16`) | `megatron/core/fusions/fused_mla_yarn_rope_apply.py` 将前向 MLA Query RoPE 从跨步 `0::2` / `1::2` 两次离散 `tl.load` 改为连续块加载，减少 `50%` warp 内存事务并保证 `sbhd`/`thd` 确定性重放。 |
| `megatron-mfsdp-v2-mcore-adapter-enable-mxfp8-2026` | `NVIDIA/Megatron-LM@7f8dcee979` (`2026-09-24`) | `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` 支持 MFSDP v2 下 MXFP8 计算与 `fp8_param` + `fp8_param_gather` 通信（相比 BF16 参数 AllGather 压缩 `2.0x`，GB200 单节点 4 GPU Step-1 基准 loss `10.899849891662598`）。 |
| `torchtitan-moe-skip-per-expert-token-count-unused-2026` | `pytorch/torchtitan@72a3a126fe` (`2026-10-27`) | `torchtitan/models/muse_spark/moe.py` 等在不读取 `num_tokens_per_expert_E` 的后端中直接传递布尔 `routing_map_TE`，跳过每层 `torch.bincount` 同步（60 层模型每步消除约 `870 us` 额外开销）。 |
| `torchtitan-rl-preserve-per-group-kv-cache-policy-update-2026` | `pytorch/torchtitan@ab51d1311d` (`2026-10-01`) | `torchtitan/rl/controller.py` 允许在 `generator_router.hot_swap=True` 或按 `policy_version` 注入 `cache_salt` 时跨策略权重更新保留 per-group KV Cache，同时对 `hot_swap=False && !reset_prefix_cache_on_weight_sync` 保持 fail-closed 拦截。 |
| `mooncake-tent-add-memfabric-transport-2026` | `kvcache-ai/Mooncake@129c9614de` (`2026-10-16`) | `mooncake-transfer-engine/tent/src/transport/memfabric` 新增基于华为 `smem_trans` 共享内存框架的 TENT `MemFabric` 跨节点/跨设备批量传输插件及无 NPU 依赖的 `memfabric-mock` CI 构建。 |
| `mooncake-store-allow-dfs-only-replica-placement-2026` | `kvcache-ai/Mooncake@c6f1d08e1c` (`2026-10-19`) | `mooncake-store/src/master_service.cpp` 移除 `dfs_replica_num > 0 && replica_num == 0` 报错，正式支持 `replica_num = 0, nof_replica_num = 0, dfs_replica_num = 1` 的纯 DFS 副本放置模式。 |
| `lmcache-coordinator-serve-cache-delete-eviction-controller-2026` | `LMCache/LMCache@d05401790f` (`2026-10-22`) | `lmcache/v1/mp_coordinator/controllers/eviction_controller.py` 将 `/cache/delete` 归属至 eviction controller（与 `/quota`、`/cache/pins` 同生命周期），确保删除请求自动保留已 pin 的 KV 键。 |
| `nixl-hf3fs-plugin-fix-atomic-races-2026` | `ai-dynamo/nixl@4ef0a4d9be` (`2026-10-12`) | `src/plugins/hf3fs/hf3fs_backend.{h,cpp}` 将 `stop_thread`、`error_status` 与 `submitted_ios` 改为 `std::atomic`（`memory_order_relaxed`），消除 HF3FS 异步 I/O 线程与提交线程的数据竞争。 |

## 2. 复现命令

```bash
python experiments/te_fp8_quantizer_megatron_mla_rope_mfsdp_mxfp8_mooncake_phase114/run.py --check
```
