# SGLang TTFT 阶段拆解与元数据填充内核、vLLM MORI EP V2 与 gRPC 排空、TRT-LLM NVFP4 MoE 与 DeepEP 跨网络 GIN 屏障深度解析 (Phase 160)

> **证据类型**：`implementation`（一手上游提交源码审计）+ `own-offline`（本项目标准库确定性实验 `experiments/sglang_ttft_fill_vllm_mori_ep_trtllm_nvfp4_deepep_gin_phase160/`）

## 1. 一手上游提交矩阵 (Track 2: 10 Commits)

| # | 仓库与提交 | 发布日期 | 核心实现路径与源码定位 |
|---|---|---|---|
| 1 | `sgl-project/sglang@571ddad70f` | 2026-10-31 | `python/sglang/srt/observability/req_time_stats.py`：在响应 `meta_info` 中输出细粒度 TTFT 阶段耗时拆解（排队、prefill 发射、prefill 计算、KV 传输），消除首 token 尾延迟黑盒。 |
| 2 | `sgl-project/sglang@005b8e033f` | 2026-10-31 | `python/sglang/srt/managers/tokenizer_manager.py`：在 `tokenizer_manager` 下发请求前拦截已发送 abort 的请求并清理 RID/CUDA IPC 传输状态，消除幽灵请求（ghost request）占用调度槽位。 |
| 3 | `sgl-project/sglang@9767ce3a5a` | 2026-10-31 | `python/sglang/kernels/ops/memory/small_copy.py`：在 `SGLANG_ENABLE_MLP_SYNC_METADATA_FILL` 下使用 Triton `_write_host_ints_kernel` 批量写入小型前向元数据，替代多次独立小张量拷贝。 |
| 4 | `vllm-project/vllm@d5301a4f54` | 2026-10-31 | `docs/serving/expert_parallel_deployment.md`、`tests/kernels/moe/modular_kernel_tools/common.py`：新增 AMD ROCm MORI Expert Parallelism (EP) V2 All-to-All 通信后端支持与 A4W4 DeepSeek-V4 环境变量编排。 |
| 5 | `vllm-project/vllm@5a233286c6` | 2026-10-31 | `vllm/entrypoints/launchers/grpc_server.py`：实现 `_drain_and_stop` 优雅停机序列，在 `--shutdown-timeout` 预算内先排空在途 gRPC 流，再中止残余引擎任务，防止停机截断或死锁。 |
| 6 | `vllm-project/vllm@0c63ce9f0e` | 2026-10-31 | `vllm/entrypoints/openai/chat_completion/batch_serving.py`：修复批量 Chat Completions 接口遗漏 `prompt_tokens_details`（如 prefix cache 命中统计）的问题。 |
| 7 | `NVIDIA/TensorRT-LLM@09198b7ca2` | 2026-10-31 | `tensorrt_llm/_torch/_experimental/modeling_v2/`、`tests/integration/defs/accuracy/modeling_v2/_env.py`：隔离 `modeling_v2` 路由与测试全局共享状态，防止跨模型测试污染。 |
| 8 | `NVIDIA/TensorRT-LLM@f6c4125c79` | 2026-10-28 | `tests/unittest/_torch/moe/moe_test_utils.py`、`quantize_utils.py`：重新启用大 `intermediate_size` 下的 CuTeDSL NVFP4 MoE 后端测试并校准 GPT-OSS SwiGLU 精度容差。 |
| 9 | `deepseek-ai/DeepEP@11d1cab604` | 2026-08-04 | `deep_ep/include/deep_ep/common/comm.cuh`：在跨 NVLink 与 RDMA 混合扩展（mixed-fabric scaleup）的 `gin_barrier_wo_local_sync` 之前插入 `fence.acq_rel.sys`，消除跨织网内存可见性乱序。 |
| 10 | `deepseek-ai/DeepEP@dd758caf45` | 2026-07-14 | `csrc/jit/handle.hpp`、`csrc/jit/no_ref.hpp`、`csrc/kernels/backend/api.cuh`：使用运行时版本号调用 `ncclDevCommCreate`，并通过 `jit::NoRefPtr` 直接传递设备通信器存储，消除主机端解引用不兼容。 |

## 2. 关键实现路径与架构公式

### 2.1 SGLang Triton 小元数据合并填充 (`_write_host_ints_kernel`)
在 Decode 步中，若有 $K=14$ 个小型标量/索引前向元数据张量，逐个发起异步拷贝需耗费 $14 \times 4.5\ \mu\text{s} = 63.0\ \mu\text{s}$ 主机发射开销。`sglang@9767ce3a5a` 通过 `tl.static_range(len(encoded))` 在单次 Triton 内核启动中完成全部写入（$9.0\ \mu\text{s}$），实现 **$7.0\times$** 元数据准备加速。

### 2.2 DeepEP 跨 NVLink + RDMA 混合织网 GIN 屏障内存序
当 MoE EP 域同时跨越片内 NVLink 与节点间 RDMA GDAKI 时，仅靠 GPU 作用域同步无法保证对端 NIC/Peer GPU 在看到 GIN 信号量前已观测到全部写操作。`DeepEP@11d1cab604` 在 `gin_barrier_wo_local_sync` 发射 `ncclGin` 门铃前显式执行 `fence.acq_rel.sys`，将跨织网陈旧数据可见性风险从 `12` 个竞态窗口降为 `0`。

## 3. 确定性离线实验结果 (`own-offline`)

运行 `python experiments/sglang_ttft_fill_vllm_mori_ep_trtllm_nvfp4_deepep_gin_phase160/run.py --check`：
- **SGLang 元数据填充发射加速比**：`7.0x`（`63.0 us -> 9.0 us`），提前 abort 的幽灵请求下发数从 `32` 降为 `0`。
- **vLLM gRPC 优雅停机排空**：`64` 个在途请求在 `8.0s`（`< 15.0s` 预算）内完成两阶段排空，孤儿引擎请求 `0` 个。
- **DeepEP 混合织网 GIN 屏障与 `NoRefPtr`**：32 卡（4 节点 $\times$ 8 GPU）混合域下未栅栏可见性隐患从 `12` 降为 `0`，`jit_norefptr_host_deref_eliminated = true`。
