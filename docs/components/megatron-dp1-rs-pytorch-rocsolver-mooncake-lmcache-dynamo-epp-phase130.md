# Megatron-LM DP=1 FP32 Reduce-Scatter 快径、PyTorch ROCm `syevd` 缓存分配器工作区、Mooncake 多节点 `tebench` 与 Store-RS、LMCache 停止事件循环保护及 Dynamo EPP（Phase 130）

> **English Edition**: [megatron-dp1-rs-pytorch-rocsolver-mooncake-lmcache-dynamo-epp-phase130.en.md](megatron-dp1-rs-pytorch-rocsolver-mooncake-lmcache-dynamo-epp-phase130.en.md)  
> **Phase 130 (`priority: 139`) · 130 阶段 & 400 实验套件双里程碑 · Track 3（分布式训练、编译栈与分离式 KV 传输层）**  
> **证据等级**: `implementation` (一手上游提交源码) + `own-offline` (确定性 CPU 仿真，非 GPU 实测)

---

## 1. 背景与核心架构问题

在分布式训练框架（Megatron-LM、PyTorch ROCm/TorchDynamo）与分离式 KV/路由基础设施（Mooncake、LMCache、NVIDIA Dynamo）中，2026-10-28 的 10 项上游提交解决了五个关键工程缺陷：

1. **Megatron-LM `DP size = 1` 下的冗余 FP32 归约与 All-to-All**：在纯 TP/PP/CP 或 Expert Parallel 分支（`world_size == 1` 的单 rank 数据并行组）中，`reduce_scatter_with_fp32_accumulation` 仍会分配完整的 FP32 临时缓冲区（如 1.34 亿元素即 **512 MiB**）、将输入从 BF16 复制为 FP32、发起单 rank `all_to_all` 并再拷回输出张量。
2. **PyTorch ROCm 批量 `eigh` (`rocsolver_*syevd_strided_batched`) 的 `hipMalloc` 假性 OOM 与 TorchDynamo `functools.wraps` 名称丢失**：`rocSOLVER` 默认用底层 `hipMalloc` 增长工作区，当 PyTorch `CUDACachingAllocator` 已持有绝大部分显存时，即使缓存池中有逾 1 GiB 空闲块，底层 `hipMalloc` 也会直接报 `rocblas_status_memory_error`；同时 TorchDynamo `UserFunctionVariable.get_name()` 读取 `co_name` 而非 `self.fn.__name__`，导致 `functools.wraps` 装饰函数的名称在编译图中丢失（2.15 RC 回归问题 `#201123`）。
3. **Mooncake 多节点 `tebench` 编排、Store-RS 零拷贝地址重叠 (`ERR_ADDRESS_OVERLAPPED`) 与 Intel XPU TENT 支持**：Store-RS 的本地 scratch pool 已在 TransferEngine 注册，若对其中切片再次注册零拷贝缓冲区会触发 `ERR_ADDRESS_OVERLAPPED`；同时 `tebench` 需要支持跨节点 SSH 屏障编排（`fanout` / `incast` / `m2n` / `all_to_all`）以及 Intel XPU 设备显存 (`--backend=tent --seg_type=VRAM`)。
4. **LMCache 重复 `close()` 向已停止事件循环投递协程导致的 11 秒挂起**：`PDBackendAsync.close()` 在 `_recv_loop` 已停止后再次调用时，仍通过 `run_coroutine_threadsafe` 投递未决分配排空协程，导致阻塞等待 `shutdown_timeout_sec + 1`（11 秒）并抛出未等待协程警告。
5. **NVIDIA Dynamo 网关 EPP (`ext-proc`) 协议过度约束与 Prefill Worker 识别脆弱性**：EPP 仅在收到完整请求体（完成分词、LoRA 改写与 KV 路由选择）后才构造 `RequestHeaders` 响应，因此无需强制 Envoy 开启 `send_body_without_waiting_for_header_response`；同时应通过显式 `WorkerType::Prefill` 元数据而非名称启发式识别分离式 Prefill 节点，并通过 Rust 类型自动生成 Python 多媒体协议模型。

---

## 2. 一手源码证据与机制深挖（10 项上游提交）

| # | 仓库与提交 | 日期 | 核心机制与架构量化指标 |
|---|---|---|---|
| 1 | `NVIDIA/Megatron-LM@79f5594` (`#8449`) | 2026-10-28 | `reduce_scatter_with_fp32_accumulation` 在 `world_size == 1` (`DP=1`) 时直接执行 `output_tensor.copy_(input_tensor)` 并返回同步 `_CompletedWorkHandle`，消除冗余的单 rank `all_to_all` 与 **512 MiB**（128M 元素）FP32 临时显存分配。 |
| 2 | `pytorch/pytorch@b15a13a` (`#199196`) | 2026-10-28 | ROCm `apply_syevd_batched_rocsolver` 通过 `rocblas_start_device_memory_size_query` 查询所需字节数，使用 PyTorch `getCUDADeviceAllocator()` 分配工作区并传入 `rocblas_set_workspace`，在释放前重置为 `(nullptr, 0)`，避免缓存分配器满载时 `hipMalloc` 触发 `rocblas_status_memory_error`。 |
| 3 | `pytorch/pytorch@3ce5d60` (`#201518`) | 2026-10-28 | 在 `torch/_dynamo/variables/functions.py` 的 `UserFunctionVariable` 上重写 `get_name()` / `get_qualname()` 以读取 `self.fn.__name__` / `self.fn.__qualname__`，修复 `functools.wraps` 装饰器在 `torch.compile` 下的名称丢失回归。 |
| 4 | `kvcache-ai/Mooncake@78f9eb6` (`#4649`) | 2026-10-28 | 新增 `benchmark/scripts/run_tebench_multinode.sh` 多节点编排器，支持 `fanout`、`incast`、`m2n`、`all_to_all` 四种拓扑、启动屏障、`write_seed`/`read_verify` 一致性校验阶段以及多节点 JSONL 汇总。 |
| 5 | `kvcache-ai/Mooncake@e03a7bf` (`#4899`) | 2026-10-28 | `store_kv_bench.py` 新增 `--store-backend`（`cpp` 与 `store-rs` 统一评测），并为 Store-RS 零拷贝缓冲区分配独立注册内存区域，消除对已注册 scratch pool 切片重复注册引发的 `ERR_ADDRESS_OVERLAPPED`。 |
| 6 | `kvcache-ai/Mooncake@047a215` (`#4901`) | 2026-10-28 | TENT `tebench` 在 `USE_XPU=ON` (`CXX=icpx`) 下支持 `--backend=tent --seg_type=VRAM` 分配 Intel XPU 设备显存，并通过 `Platform::copy` 支持 XPU VRAM 与 `file://` 目标段的直传基准测试。 |
| 7 | `LMCache/LMCache@abc685b` (`#6116`) | 2026-10-28 | `PDBackendAsync.close()` 在 `_recv_loop` 已停止时跳过 pending-allocation drain 投递，消除重复 `close()` 导致的 `shutdown_timeout_sec + 1`（**11.0 秒**）阻塞挂起与未等待协程告警。 |
| 8 | `ai-dynamo/dynamo@8a9517c` (`#15605`) | 2026-10-28 | 新增 `scripts/generate_media_protocols.py`，从 Rust 权威媒体类型生成 JSON Schema 与 Python `audio_protocol.py`、`image_protocol.py`、`video_protocol.py`，消除跨语言协议漂移。 |
| 9 | `ai-dynamo/dynamo@9937486` (`#17125`) | 2026-10-28 | Dynamo 推理网关 `ext-proc` (`server.rs`) 移除对 `send_body_without_waiting_for_header_response` 标志的强制要求，仅验证双向 `FULL_DUPLEX_STREAMED`，因为 `RequestHeaders` 响应本来就在接收完整请求体并选定 `x-dynamo-worker-instance-id` 后才发出。 |
| 10 | `ai-dynamo/dynamo@9087efc` (`#17325`) | 2026-10-28 | Dynamo EPP (`deploy/inference-gateway/ext-proc/src/epp.rs`) 使用显式 `WorkerType::Prefill` 识别分离式 Prefill Worker，替代脆弱的名称/端点字符串匹配。 |

---

## 3. 确定性实验验证与架构师指南

`experiments/megatron_dp1_rs_pytorch_rocsolver_mooncake_lmcache_dynamo_epp_phase130/run.py --check` 验证了全部 5 项核心不变量：
- `megatron_dp1_eliminates_512mib_fp32_buffer = true`
- `pytorch_rocsolver_caching_allocator_avoids_oom = true`
- `mooncake_store_rs_avoids_err_address_overlapped = true`
- `lmcache_repeated_close_eliminates_11s_hang = true`
- `dynamo_epp_uses_worker_type_and_full_duplex = true`

### 架构师决策建议与可证伪问题
- **RDMA 内存池注册边界**：在使用 Mooncake TransferEngine + Store-RS 时，严禁将已整体注册（`registerLocalMemory`）的 scratch pool 的子切片再次作为独立零拷贝 buffer 注册；必须复用父注册区或分配独立虚拟地址段以避免 `ERR_ADDRESS_OVERLAPPED`。
- **可证伪后续问题**：在 Mooncake 多节点 `tebench` 的 `incast` 与 `all_to_all` 模式下，当并发 Initiator 节点数从 4 增至 32 时，C++ Store 与 Rust `Store-RS` 在元数据路由延迟与 RDMA 尾延迟（P99）上的分界点位于多少 QPS？
