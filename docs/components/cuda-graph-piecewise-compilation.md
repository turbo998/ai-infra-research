# 分段 CUDA Graph 编译池（`PiecewiseBackend` 与 `BreakableCUDAGraphCapture`）：CPU 发射墙、显存池复用与动态 Padding 浪费边界

> **证据类型声明**：本文源码分析属 `implementation`（固定提交的 `vllm-project/vllm` 与 `sgl-project/sglang` 一手源码），算术及桶对齐仿真属 `own-offline`（[`experiments/cuda_graph_piecewise_capture/run.py`](../../experiments/cuda_graph_piecewise_capture/run.py)），不宣称物理 GPU 实测（`own-hardware`）。

---

## 1. 为什么在 H100 / B200 上 Host CPU 算子发射墙会吞噬 50% 以上的 TPOT？

在 DeepSeek-V3 / Qwen3-MoE 等 61 层大模型（`TP=8`）的 Decode 或小块 Chunked Prefill 阶段，每一层包含 RMSNorm、QKV/O 投影、RoPE、Attention、MoE 门控、Dispatcher/Combine 与 AllReduce 等约 `8` 个细粒度算子，单步总计约 **`488` 个 CUDA Kernel 发射**。
- Python / PyTorch Dispatcher + `cudaLaunchKernel` 的单算子主机侧开销约为 **`4.1 us/op`**，单步累计 CPU 发射耗时达 **`2.001 ms/step`**；
- 在 Blackwell `B200 NVFP4`（`8.0 TB/s` HBM，`4.5 PFLOPS`）上，当 `BS=1` 时 GPU 实际计算与访存仅需 **`1.575 ms`**。若采用 Eager 逐算子发射，主机侧未掩盖的 `1.449 ms` CPU 发射气泡会将端到端 TPOT 推高至 **`3.025 ms`**（吞噬 **`47.9%`** 端到端延迟，吞吐损失近半）。

---

## 2. 一手源码机制：vLLM `PiecewiseBackend` vs `BreakableCUDAGraphCapture` vs SGLang `CudaGraphConfig`

### 2.1 FX 图切分与单尺寸具体化（`vllm/compilation/piecewise_backend.py` & `partition_rules.py`）
- **源码定位**：`vllm-project/vllm` (`commit 82a24bb82dd9344cce7a43195bb51a2390adf00a`, `published_at: 2026-10-18`)：
  - `vllm/compilation/partition_rules.py` (blob `8c2e5d6d2b3325aeaac0d7adb31aff1fe672c7e6`, lines `14-74`)：`should_split(node, splitting_ops)` 与 `inductor_partition_rule_context` 在 `splitting_ops`（默认如 `vllm::unified_attention_with_output` 等动态 KV 元数据算子）处强制打断 FX 图，将一个 61 层 Transformer 拆分为 **`62` 个静态 MLP/Norm/AllReduce 子图**与 **`61` 个 Eager Attention 调用**。
  - `vllm/compilation/piecewise_backend.py` (blob `576cd941f6a8946b93e3c1c393b98eba2a77cb12`, lines `37-76`, `86-192`, `342-379`)：`create_concrete_args(graph, size)` 将 Dynamo 捕获的 `torch.SymInt` 符号维度替换为 `compile_sizes` 中的具体桶尺寸；运行期 `_find_range_for_shape(runtime_shape)` 在 $O(1)$ 时间内分派至精确匹配的 `Range(start=size, end=size)` 或通用区间编译项。

### 2.2 全局共享内存池与弱引用输出释放（`vllm/compilation/cuda_graph.py` & `cudagraph_pool.py`）
- **源码定位**：`vllm/compilation/cuda_graph.py` (blob `2dea34e0ff4d99fd2f1d3ee9351501eeb5fc5486`, lines `145-362`) 与 `vllm/compilation/cudagraph_pool.py` (blob `c4cfeabd81ce4dba45815a598656358b0a0aec41`, lines `32-45`)：
  1. **全局单流内存池复用**：`self.graph_pool = current_platform.get_global_graph_pool()`。由于不同 Batch Bucket（如 `BS=1, 2, 4, ..., 512` 共 `32` 个桶）在同一生命周期内绝不会并发重放，所有桶与所有分段子图共享同一个 `graph_pool`；
  2. **CuMem 休眠联动**：当开启 RL 睡眠模式（`vllm_config.use_cumem_cudagraph_pool`）时，`capture_pool()` 将 CUDA Graph 内存池与 PyNCCL 图分配器 (`set_graph_pool_id(cumem_pool)`) 统一切入 `CuMemAllocator.cudagraph_pool()`，从而允许在 RL 训练阶段一键卸载 CUDA Graph 激活缓冲区；
  3. **弱引用输出防泄漏**：在 `with torch.cuda.graph(cudagraph, pool=graph_pool)` 内部（lines `326-338`），vLLM 对捕获输出执行 `output = weak_ref_tensors(output)`，立即释放 Python 强引用，使得下一层或下一个 Batch 桶的 CUDA Graph 可以原地复用同一段物理设备地址。

### 2.3 免 FX 拆图的流捕获断点机制（`vllm/compilation/breakable_cudagraph.py` & SGLang `cuda_graph_config.py`）
- **源码定位**：
  - `vllm/compilation/breakable_cudagraph.py` (blob `b70fe70d8c80458ad494d92ec394f064fde62bee`, lines `67-200`)：受 `sgl-project/sglang#19102` 启发，`BreakableCUDAGraphCapture` 不再依赖 `torch.compile` 预先把模型切成几十个 FX GraphModule，而是用单一流捕获上下文包裹完整 `model(*static_inputs)`。当遇到被 `@eager_break_during_capture` 装饰的算子时，调用 `self._current_graph.capture_end()` 结束当前图段，在捕获流上以弱引用参数 Eager 执行该算子（记录为 `self.segments.append(fn)`），随后立即调用 `_begin_segment()` 开启下一段 `CUDAGraph`。源码第 `82-96` 行明确强调：`@eager_break_during_capture` **必须位于 `@maybe_transfer_kv_layer` 最外层**，否则 PD 分离的 `wait_for_layer_load` / `save_kv_layer` 主机侧同步会被录制进图内导致重放死锁。
  - `sgl-project/sglang` (`commit 0f4aedfecc3e1e420292541af4e9da8909b9b4f7`, `python/sglang/srt/model_executor/cuda_graph_config.py`, blob `ed481b3c70c02dd4ad3064f1be499f417f9d8ad9`, lines `40-150`)：将 `Phase.DECODE` 默认设为 `Backend.FULL`，将 `Phase.PREFILL` 默认设为 `Backend.BREAKABLE` (`default_prefill_backend()`)，以兼顾 Decode 极致的整图发射与 Prefill 动态上下文长度的灵活性。

```
Eager Mode (488 ops/step, 2.001 ms CPU launch):
  [Op1][Op2]...[Op488] ---> GPU stalls on host dispatch when BS <= 16 on B200

Piecewise / Breakable CUDA Graph (62 graph segments + 61 Eager Attention breaks):
  [Graph Seg 0 (padded_bs)] -> [Eager Attn 0 (unpadded_bs!)] -> [Graph Seg 1 (padded_bs)] ...
   * Shares 1 global_graph_pool across all 32 buckets (76.55 GiB -> 1.04 GiB, 73.63x savings)
   * Avoids padding KV Cache reads in Attention on unaligned batch sizes (e.g. BS=33 -> 40)
```

---

## 3. 离线实验结果（`experiments/cuda_graph_piecewise_capture/`）

| 指标 / 场景 | Eager 逐算子发射 | Piecewise / Breakable 图 | Full CUDA Graph | 关键架构发现 |
|---|---|---|---|---|
| 单步 Host CPU 发射耗时 | `2.001 ms` (`488` ops) | `0.399 ms` (`62` 段 + `61` Eager) | `0.018 ms` (单图 replay) | 分段图消除 `80.1%` 主机发射开销 |
| 32 个捕获桶 HBM 占用 (`BS=1..512`) | `0.38 GiB` (仅静态 I/O) | **`1.04 GiB`** (`global_graph_pool` 共享) | `1.04 GiB` (共享池) / **`76.55 GiB`** (未共享池) | 全局共享池 + `weak_ref_tensors` 带来 **`73.63x`** 显存压缩 |
| `B200 NVFP4` `BS=1` 端到端 TPOT | `3.025 ms` | **`1.575 ms` (`1.92x` 提速)** | `1.593 ms` (`1.90x` 提速) | 低并发下彻底打穿 CPU Launch Wall |
| `H100 FP8` 非对齐 `BS=33`（向上对齐至 `40`） | `7.136 ms` | **`7.165 ms`** | `7.954 ms` (`0.90x` 退化) | Full Graph 强制对 7 个 Dummy Slot 执行 KV 读取，比 Piecewise 慢 `0.789 ms` |
| `B200 NVFP4` 非对齐 `BS=257`（对齐至 `288`） | `14.507 ms` | **`14.786 ms`** | `16.232 ms` (`0.89x` 退化) | 大 Batch 非对齐点处，Piecewise 避免 `31` 个 Dummy 请求的 KV 带宽浪费，快 **`1.446 ms`** |

---

## 4. 架构师设计准则与可证伪问题

1. **何时选择 `FULL` vs `PIECEWISE/BREAKABLE`？**
   - 当 Attention 后端支持静态 Paged KV 元数据更新且工作负载以中小 Batch Decode（`BS <= 64`）为主时，`Backend.FULL` 具有最低的 CPU 发射开销（`0.018 ms`）；
   - 当工作负载包含 **Chunked Prefill、Hybrid Mamba/Attention、PD 逐层 KV 传输（`@maybe_transfer_kv_layer`）或高并发非对齐 Batch（`BS > 64`）** 时，`PiecewiseBackend` / `BreakableCUDAGraphCapture` 是更优解——它不仅免去了为每个上下文长度桶录制整图的显存爆炸，还避免了在非对齐 Batch 上为 Dummy Padding Token 空读数十 MB 的 KV Cache。
2. **可证伪实验问题**：在开启 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 时，对比 FX `PiecewiseBackend` 与流捕获断点 `BreakableCUDAGraphCapture` 的冷启动编译时间（预计从数分钟降至 `<15 s`）。
