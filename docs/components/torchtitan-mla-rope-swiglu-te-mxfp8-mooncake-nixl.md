# TorchTitan 行内 Int32 对 MLA RoPE 旋转与无掩码 SwiGLU 反向、TransformerEngine 逐 Tile MXFP8 缩放重排、Megatron 异步卸载暂存及 Mooncake/LMCache/NIXL/Dynamo 传输状态机防护

> **English Edition**: [torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl.en.md](torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl.en.md)  
> **Phase 63 · Priority 72 · Track C (`question-204`)** · 核验日：`2026-10-24`（UTC+08:00）

---

## 1. 核心架构问题与背景

在 DeepSeek-V3（671B）/ Kimi K2.5 的 `torch.compile` 区域化编译、Blackwell MXFP8 缩放因子重排（Scale Swizzling）、动态多模态推理引擎以及 RDMA/UCX 远程 KV 传输平面中，近期暴露出五项核心工程瓶颈：

1. **TorchTitan MLA RoPE 跨步切片视图（Strided View）与 SwiGLU 掩码拼接导致的带宽利用率塌陷**：
   - 在 DeepSeek-V3 与 Kimi K2.5 的 `_apply_rope_to_q_and_k_pe`（`mla_qk` 区域）中，查询张量 $q \in \text{BF16}[T, H, 192]$ 被切分为 `q_nope [T, H, 128]` 与 `q_pe [T, H, 64]`。由于 `q_pe` 是跨步切片（row stride `192`），Inductor 无法直接将其视为连续的 `int32` 对，只能退化为 stride-2 标量加载，导致该区域仅跑出 GB300 HBM 带宽下界（bandwidth floor）的 **59%–69%**（DSv3 64k prefill 耗时 `1380.0 us`，而带宽下界仅 `945.7 us`）。
   - 在 `SwiGLU` 编译区域反向传播中，Inductor 默认将打包梯度 `grad_gate_up [8T, 2, F]` 融合为单个带掩码（`tl.where`）的逐点核，在 $T=16\text{k}, F=2048$ 下反向核耗时 `788 us`（为带宽下界 `377 us` 的 **2.1 倍**）。
2. **TransformerEngine MXFP8 窄 $K$ 缩放重排核的 CTA 并行度饥饿**：旧版 MXFP8 窄 $K$（$\le 32$ 个 `128x4` 缩放 tile，即原始激活列数 $K \le 4096$）重排核将最多 `32` 个 `128x4` tile 打包进单个 `1024` 线程的 CTA。对于 $4096 \times 4096$ 的激活张量，整张 GPU 仅启动 **1 个或极少数 CTA**，SM 占用率严重不足。
3. **Megatron-LM `DynamicInferenceEngine` 主线程 H2D 阻塞与已完成请求滞留 GPU 显存**：`payload_stager.stage()` 在引擎主循环内同步创建 CUDA 张量，且已完成请求仍持有 GPU 侧 `prompt_tokens` 与 VLM `media_tensors`。
4. **Mooncake RDMA 异步事件盲目轮询与 TENT `cuda_memcpy_sync` 主机 `memcpy` 误回退崩溃**：
   - Mooncake 经典 RDMA `EndpointStore` 的 1 Hz 恢复线程此前对**任何**使上下文失活的异步事件（包括不可恢复的 `IBV_EVENT_DEVICE_FATAL`、`IBV_EVENT_CQ_ERR`、`IBV_EVENT_WQ_FATAL`）均尝试查询端口并探测 GID，违反预期的仅限端口故障恢复契约。
   - Mooncake TENT 在跨线程执行 `cuda_memcpy_sync` 时，若调用线程 `cuCtxGetCurrent(&ctx)` 为空，直接调用 `cudaMemcpy` 会失败；更危险的是上层 `DataCopy` 仅将 `TENT_DEVICE_NOT_SUPPORT` 视为设备错误，对其它错误码竟直接**在 GPU 设备指针上回退调用 CPU `std::memcpy`**，触发进程段错误（Segfault）。
5. **LMCache 周期线程竞争、NIXL UCX 发送侧观测盲区与 Dynamo NIXL 首请求延迟尖峰**：LMCache `PeriodicThread.stop()` 无锁并发 `join()`、Mooncake 批量查询标签污染；NIXL UCX 缺少 sender 侧分阶段追踪；Dynamo 延迟在首个推理请求时才初始化 NIXL connector。

---

## 2. 上游源码实现路径与关键机制

### 2.1 TorchTitan：行内 `int32` 对 MLA RoPE 旋转（`2fddc6b7a0` / `#5529`, `21db340cba` / `#5161`）与无掩码 SwiGLU（`eb48bab350` / `#5460`）

1. **`_RotateTail` 行内 `int32` 对旋转（`#5529`）**：
   - 先由 `#5161`（`21db340cba`）将 $q$ 的 split $\to$ RoPE $\to$ cat 与 `k_pe` RoPE 封装为局部编译区域 `mla_qk`（并开启 `use_contiguous_tensor_cat=True` 消除 `CatArrayBatchedCopy`，将 DSv3 671B 16k 双层 fwd+bwd 步耗时从 `182.44 ms` 降至 **`178.58 ms`**，`-2.1%`）。
   - 随后 `#5529`（`2fddc6b7a0`）在 `ComplexRoPE` 且 $q$ 为连续 BF16 张量时，通过 `_RotateTail` 直接将整行 `q [T, H, 192]` 视图转换为 `int32 [T, H, 96]`（每个 `int32` 词打包一对 BF16 实部/虚部），对末尾 `32` 个 `int32` 词（对应 `q_pe` 的 `64` 维）调用 `_RotatePairs.rotate`，再与前 `64` 个未修改的 `int32` 词（`q_nope` 的 `128` 维）拼接并转回 BF16。

| 模型与模式（GB300, CUDA Graphs） | 序列长度 $T$ | 旧版耗时 ($\mu\text{s}$) | `#5529` 耗时 ($\mu\text{s}$) | 延迟降幅 / 加速比 | HBM 带宽下界 ($\mu\text{s}$) | 下界带宽利用率 |
|---|---:|---:|---:|---|---:|---:|
| **DeepSeek-V3 671B** · `train` | `16,384` | `761.3` | **`492.8`** | **-35.3%** (**1.545x**) | `472.8` | **95.9%** |
| **DeepSeek-V3 671B** · `train` | `65,536` | `2985.5` | **`1926.6`** | **-35.5%** (**1.550x**) | `1891.3` | **98.2%** |
| **DeepSeek-V3 671B** · `prefill` | `65,536` | `1380.0` | **`953.5`** | **-30.9%** (**1.447x**) | `945.7` | **99.2%** |
| **DeepSeek-V3 671B** · `prefill` | `131,072` | `2748.1` | **`1899.5`** | **-30.9%** (**1.447x**) | `1891.3` | **99.6%** |
| **Kimi K2.5** · `train` | `65,536` | `1518.6` | **`989.7`** | **-34.8%** (**1.534x**) | `950.6` | **96.0%** |
| **Kimi K2.5** · `prefill` | `131,072` | `1429.8` | **`977.9`** | **-31.6%** (**1.462x**) | `950.6` | **97.2%** |

2. **SwiGLU 反向双路独立存储（`eb48bab350` / `#5460`）**：在 SwiGLU 编译区域同时设置 `max_pointwise_cat_inputs=0` 与 `max_complex_pointwise_cat_inputs=0`（后者默认值为 `8`，若只设前者仍会将 2 输入 cat 路由至掩码核），使 Inductor 为 `grad_gate` 与 `grad_up` 分别发射独立无掩码写回。在 GB300 上 $T=16\text{k}, F=2048$ 的 SwiGLU fwd+bwd 从 `1440.0 us` 降至 **`1047.8 us`**（**-27.2% / 1.374x**），反超手写 Triton override（`1060.5 us`）。

### 2.2 TransformerEngine 逐 Tile MXFP8 缩放重排（`33e7ff3160` / `#3658`）与 Megatron 异步暂存（`894b5114e3` / `#8169`）

1. **TE `NVTE_MXFP8_PER_TILE_SWIZZLE=1` 逐 Tile CTA 调度**：在 `transformer_engine/common/util/swizzle.cu` 中，新增 `per_tile_mxfp8_scale_swizzle_kernel_rowwise`（每 block `128` 线程处理 1 个 `128x4` tile，grid 上限 `kRowwiseMaxBlocks = 1536`）与 `colwise`（每 block `256` 线程，32-row chunk 下 grid 上限 `1024`），支持单次 launch 打包最多 `kMaxTensorsPerKernel = 64` 个张量。对于 $4096 \times 4096$ 激活矩阵，CTA 数量从旧版窄 $K$ 核的 **`1` 个激增 $32\times$ 至 `32` 个**。同时 `#3540`（`1c9f6f8f39`）为 SM100/SM110 FA4 解锁 $D=256$ 前向与 `b31`/`b33` THD `seqused_q/k` 兼容。
2. **Megatron-LM `DynamicInferenceEngine` 异步暂存与显存零残留（`894b5114e3` / `#8169`）**：将 `payload_stager.stage()` 卸载至后台线程池（通过 `_staging_tasks` 与 `_staging_tasks_peak` 追踪队列水位），并在 `OffloadedRequestPayload.from_request` 中将 `prompt_tokens` 与 VLM `media_tensors` 立即 detach 迁回 CPU，使已完成请求占用的 GPU 显存降至 **`0 B`**。此外 `#8010`（`fa322ce620`）修复 meta device 共享词表权重在 `all_reduce` 前的物化，`#3698`（`0156c21373`）补齐 `MultiLatentAttention` 自定义 `config.softmax_scale` 的 `mscale * mscale` 缩放。

### 2.3 Mooncake RDMA/TENT 恢复、LMCache 线程安全与 NIXL/Dynamo 传输生命周期

1. **Mooncake RDMA 1 Hz 端口恢复严格门控（`8c19b7286c` / `#4715`）**：在 `rdma_context.cpp` 中新增原子字段 `context_disabling_event_`，仅当使上下文失活的事件确认为 `IBV_EVENT_PORT_ERR` 且未处于熔断窗口时，1 Hz 轮询线程才调用 `ibv_query_port` 并在 `IBV_PORT_ACTIVE` 时执行 ` probeRdmaContext()` + `deleteAllEndpoint()` 恢复；对 `DEVICE_FATAL` / `CQ_ERR` / `WQ_FATAL` 严格禁止盲目重试。
2. **Mooncake TENT `CUcontext` 保护与 `TENT_DEVICE_PLUGIN_ERROR`（`b2c7d0a7b5` / `#4839`）**：在 `cuda_plugin.cpp` 的 `cuda_memcpy_sync` 中通过 `cuCtxGetCurrent` / `cuDevicePrimaryCtxRetain` / `cuCtxSetCurrent` 保证跨线程上下文有效并在退出时恢复原上下文；同时新增 `TENT_DEVICE_PLUGIN_ERROR = -3`，阻止 `DataCopy` 在插件失败时将 GPU 指针交给主机 `memcpy`。
3. **Mooncake 命名空间驱逐（`fff8e90952` / `#4811`）、LMCache 锁保护（`e0e92b8356` / `#5281`, `e45d4d5a85` / `#5449`）、NIXL UCX 发送侧追踪（`8d5660c118` / `#2443`）与 Dynamo 预热（`6278e410d2` / `#17060`）**：Mooncake 将驱逐拆分为独立的 `Evictor` 与 `OffloadingReclaimer` 并支持 namespace 级配额；LMCache 为 `PeriodicThread` 加锁并修复 `MooncakeLookupClient` 批量 `exists` 的 `request_configs` 透传；NIXL UCX 插件埋点 `nixl::submit`、`nixl::wire.submitted`、`nixl::notif.sent` 三阶段；Dynamo 在引擎启动期主动调用 `nixl_connector.warmup()` 消除首请求冷启动抖动。

---

## 3. 本项目确定性实验验证与架构师设计建议

验证脚本：`experiments/torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl/run.py`（单元测试：`tests/test_phase63_experiments.py`）：
- 在 GB300 上编译 DeepSeek-V3 / Kimi K2.5 时，启用 `mla_qk` 行内 `int32` 对旋转与无掩码 SwiGLU 区域选项（`max_pointwise_cat_inputs=0, max_complex_pointwise_cat_inputs=0`），可直接逼近 **95.9%–99.6%** HBM 带宽下界并消除对外部手写 Triton 核的维护依赖。
- 在多租户 RDMA KV 传输集群中，必须区分可恢复链路抖动（`IBV_EVENT_PORT_ERR`）与致命硬件/队列错误（`DEVICE_FATAL` / `CQ_ERR`），并在 CUDA Driver/Runtime 混合线程池中显式绑定 Primary Context。
