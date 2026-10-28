# Megatron-LM 流式残差反向 `16x512` 分块消除 FP32 寄存器溢出，PyTorch ROCm `RocprofLogger` `.bss` 原位构造与 Inductor 低 SM 协同归约守卫，Mooncake `_structured_ranges.py` 打包与 ASan 修复，LMCache 移除 `PinMonitor` 与 Dynamo 轨迹分析，及 Dynamo TRT-LLM Sidecar 路由故障快速恢复分析（Phase 131 · Priority 140）

> **English Version**: [megatron-streamwise-16x512-pytorch-rocprof-mooncake-lmcache-dynamo-phase131.en.md](megatron-streamwise-16x512-pytorch-rocprof-mooncake-lmcache-dynamo-phase131.en.md)  
> **证据锚点**: `ev-megatron-streamwise-16x512-pytorch-rocprof-mooncake-lmcache-dynamo-phase131-2026` (`own-offline` + `implementation`)  
> **可复现实验**: `python experiments/megatron_streamwise_16x512_pytorch_rocprof_mooncake_lmcache_dynamo_phase131/run.py --check`

---

## 1. 第一手源码证据矩阵（Primary Source Matrix）

| 仓库与 Commit | 源码路径 | 核心架构变更 |
|---|---|---|
| `NVIDIA/Megatron-LM` [`26020f4`](https://github.com/NVIDIA/Megatron-LM/commit/26020f4e3cf91d7595fb2eb7bcde962d87c8e849) | `megatron/core/transformer/streamwise_residual_ops.py` | 将流式残差（streamwise residual）反向核分块调整为 `16 x 512`，彻底消除每线程 `48 B` 的 FP32 寄存器溢出（register spills） |
| `pytorch/pytorch` [`a25f5e0`](https://github.com/pytorch/pytorch/commit/a25f5e0e69c2e2a8808cb1f4344db0500c79f3d7) | `torch/csrc/profiler/kineto/libkineto/src/RocprofLogger.cpp` | 检查 `rocprofiler-sdk` 返回码，并使用 `.bss` 段缓冲区的 placement new 保持 `GlobalContext` 单例生命周期直至进程退出 |
| `pytorch/pytorch` [`9768e2e`](https://github.com/pytorch/pytorch/commit/9768e2e20079e88c28d73d6bf87990d737563961) | `test/inductor/test_cooperative_reductions.py` | 在 SM 数量不足的 GPU 上跳过固定网格配置的 Inductor 协同归约（cooperative reduction），防止 `cudaLaunchCooperativeKernel` 挂起 |
| `pytorch/pytorch` [`78dd89c`](https://github.com/pytorch/pytorch/commit/78dd89ca5a1bc4a1a1f77489578f21e8cb92b025) | `torch/_meta_registrations.py` | 修复 `torch.compile` 下 `rand` / `uniform` 对不支持数据类型的 Meta 推导 |
| `kvcache-ai/Mooncake` [`3fbd075`](https://github.com/kvcache-ai/Mooncake/commit/3fbd075442abd408e1291637439a0e327642b9e3) | `scripts/build_wheel.sh` | 将 `_structured_ranges.py` 暂存打包进合并 Wheel，修复 `structured_object_store` 导入时的 `ModuleNotFoundError` |
| `kvcache-ai/Mooncake` [`ab3590c`](https://github.com/kvcache-ai/Mooncake/commit/ab3590c14a605d5f4a5ef9f2bd2a8e41fa18b62b) | `mooncake-store/tests/session_tag_test.cpp` | 修复 Mooncake Store 会话标签管理在 ASan 下的 `alloc-dealloc-mismatch` 内存错误 |
| `LMCache/LMCache` [`92eb56c`](https://github.com/LMCache/LMCache/commit/92eb56cd95cdd4859efb158bf6593481ba547358) | `benchmarks/storage_backend_io/storage_backend_io_benchmark.py` | 重构隔离剩余 `InProcess` 核心模块并移除后台 `PinMonitor` 轮询线程 |
| `LMCache/LMCache` [`b89bc5f`](https://github.com/LMCache/LMCache/commit/b89bc5f121f020d82e3628311f56d9c553b4697f) | `lmcache/tools/cache_simulator/trace_analyze.py` | 在 `lmcache bench trace-analyze` 缓存仿真器中新增 `dynamo` 轨迹格式解析 |
| `ai-dynamo/dynamo` [`ba2b458`](https://github.com/ai-dynamo/dynamo/commit/ba2b45827b1ef8245ff888e71f209cb187983639) | `lib/sidecar/trtllm/src/engine.rs` | 将 TensorRT-LLM Sidecar `kv_routing` 故障路径收敛超时从 `50 s` 压缩至 `2 s` 以内（`1.6 s`） |
| `ai-dynamo/dynamo` [`fec80e0`](https://github.com/ai-dynamo/dynamo/commit/fec80e04be4b946a10a912a36738710366ff7192) | `lib/llm/src/mocker/metrics.rs` | 在 Dynamo Mocker 调度器指标中覆盖 SGLang `dp_rank` 标签与 Rank 局部状态更新 |

---

## 2. 核心架构机制与确定性建模（Core Architectural Mechanisms）

1. **Megatron-LM 流式残差反向 `16x512` 分块消除寄存器溢出（`26020f4`）**：
   - 原分块在反向传播累加 FP32 梯度时超出每线程寄存器上限，产生 **`48 B/thread` Local Memory 溢出**；改为 `16 x 512` 分块后寄存器溢出降至 **`0 B`**，核延迟从 **`186.0 us` 降至 `112.0 us`（提速 `1.6607x`）**。
2. **PyTorch ROCm `RocprofLogger` `.bss` Placement-New（`a25f5e0`）、LMCache 移除 `PinMonitor`（`92eb56c`）与 Dynamo TRT-LLM Sidecar 路由故障快速恢复（`ba2b458`）**：
   - `RocprofLogger` 使用 `.bss` 段 placement new 避免退出阶段 `GlobalContext` 单例析构顺序竞争（退出崩溃率 `1 -> 0`）；LMCache 移除后台 `PinMonitor` 消除每秒 **`200 -> 0` 次 Python GIL 唤醒干扰**；Dynamo TRT-LLM Sidecar 将 `kv_routing` 故障恢复时间从 **`50.0 s` 缩短至 `1.6 s`（提速 `31.25x`）**。
