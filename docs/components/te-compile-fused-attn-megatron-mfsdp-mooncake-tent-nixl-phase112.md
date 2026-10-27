# Phase 112 Track C：TransformerEngine FusedAttention `torch.compile` 与确定性 dGLU、Megatron-LM MFSDP Unshard/MXFP8 事件、TorchTitan EMA 重播、Mooncake TENT 上下文恢复与 NIXL POSIX 回滚（2026-10-28）

> **英文对照 / English Version**: [`te-compile-fused-attn-megatron-mfsdp-mooncake-tent-nixl-phase112.en.md`](./te-compile-fused-attn-megatron-mfsdp-mooncake-tent-nixl-phase112.en.md)  
> **7-Card 4:5 视觉评测报告**: [`../reports/ai-infra-phase112-benchmark-cards.html`](../reports/ai-infra-phase112-benchmark-cards.html)  
> **可复现实验目录**: [`../../experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/`](../../experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/)

---

## 1. 证据分级与一手代码定位（10 项分布式训练、KV 存储与网络传输提交）

| # | 仓库与 Commit | 日期 | 证据等级 | 关键代码路径与核心修复/优化机制 |
|---|---|---|---|---|
| 1 | [`TransformerEngine@7e7898fa82`](https://github.com/NVIDIA/TransformerEngine/commit/7e7898fa82719ebc1069cfa29b9ccd3b6eec4a24) | 2026-10-08 | `implementation` | `transformer_engine/pytorch/attention/dot_product_attention/backends.py`、`cpp_custom_ops.py`：新增 `te_fused_attn_fwd` / `te_fused_attn_bwd` 扁平张量/标量签名自定义算子与 `register_fake` 元内核，支持 `torch.compile(fullgraph=True)` 零图断裂捕获 `FusedAttention`（BF16/FP16 及非 `DelayedScaling` FP8） |
| 2 | [`TransformerEngine@c25fa27525`](https://github.com/NVIDIA/TransformerEngine/commit/c25fa275259808f6f6cc4fa0b8b285cfd453287a) | 2026-10-08 | `implementation` | `transformer_engine/common/fused_glu/fused_glu.cu`、`pytorch/cpp_extensions/fused_attn.py`：在 `nvte_ACT_and_dACT_and_mul_dprob` 中新增 2D/3D (chunked) workspace 先写出每行 partial `dprob` 再显式归约的确定性路径，解除 `torch.use_deterministic_algorithms(True)` 对融合分组 MLP 的阻塞 |
| 3 | [`Megatron-LM@17fc7ba109`](https://github.com/NVIDIA/Megatron-LM/commit/17fc7ba10998cce2d2dba210e3a70eb09d604651) | 2026-10-27 | `implementation` | `megatron/core/distributed/fsdp/src/megatron_fsdp/{megatron_fsdp,param_and_grad_buffer}.py`：将 `_ParameterGroup.model_weight_buffer_all_gather_finished` 作为 unshard 就绪唯一判据，并为 MXFP8 rowwise/colwise 双缓冲分配独立 CUDA Event (`all_gather_event`, `all_gather_colwise_event`) |
| 4 | [`Megatron-LM@74c9420e35`](https://github.com/NVIDIA/Megatron-LM/commit/74c9420e35cc3c1226e53421f5ff718653aa8399) | 2026-10-19 | `implementation` | `tests/unit_tests/dist_checkpointing/models/test_mfsdp_Checkpoint.py`：将并行状态初始化与销毁纳入 `try...finally`，确保单元测试失败时仍可靠清理并行组与临时目录 |
| 5 | [`torchtitan@ba53205255`](https://github.com/pytorch/torchtitan/commit/ba53205255cc89e0479096cd061fccbb71e30112) | 2026-10-17 | `implementation` | `torchtitan/components/checkpoint.py`：新增 `EMAStateAdapter` 将 `ema.reseed_from_model()` 绑定在 `load_state_dict` 入口，修复非 DCP 后端、`--checkpoint.model-only` 或 `--checkpoint.exclude-from-loading=ema` 加载后 EMA 残留随机初始化权重的问题 |
| 6 | [`Mooncake@edd323921d`](https://github.com/kvcache-ai/Mooncake/commit/edd323921d531d703f825bd042be552c878151a3) | 2026-10-22 | `implementation` | `mooncake-store-rs/src/{rpc_service,tenant_context}.rs`：在 Rust Store RPC 分发前强制校验租户控制面绑定 (`validate_tenant_binding`)，未授权跨租户请求直接拒绝并返回 `-3250` (`RPC_UNAUTHENTICATED`) |
| 7 | [`Mooncake@e3e9d70eea`](https://github.com/kvcache-ai/Mooncake/commit/e3e9d70eea3c8ea9eb236b4c9c92f361fbbe533f) | 2026-10-21 | `implementation` | `mooncake-transfer-engine/tent/include/tent/runtime/platform/cuda.h`：`CudaContextGuard` 使用 `cuCtxGetCurrent(&saved_ctx_)` 并在析构时调用 `cuCtxSetCurrent(saved_ctx_)` 精确恢复调用方非 primary CUDA context（或无上下文状态） |
| 8 | [`LMCache@e0e92b8356`](https://github.com/LMCache/LMCache/commit/e0e92b83568ab79eeee824cbbf19992ed7374f3e) | 2026-10-22 | `implementation` | `lmcache/v1/periodic_thread.py`：`PeriodicThread.start()` 检查旧 `self._thread.is_alive()`，当 `stop(timeout)` 超时未退出时拒绝启动第二个重叠后台线程并维持 `self._running = False` |
| 9 | [`dynamo@492108cd43`](https://github.com/ai-dynamo/dynamo/commit/492108cd434efc46920ba230175c98faf5c0c117) | 2026-10-27 | `implementation` | `lib/llm/src/http/service/service_v2.rs`：前端进入优雅排空 (`drain`) 拒绝新请求返回 `503 Service Unavailable` 时显式附加 `Connection: close` 响应头，迫使 HTTP/1.1 keep-alive 客户端立即重连健康副本 |
| 10 | [`nixl@f7533eb8a5`](https://github.com/ai-dynamo/nixl/commit/f7533eb8a547c4d2935cbcee6edf262eca7a78f6) | 2026-10-12 | `implementation` | `src/plugins/posix/{posix_backend.cpp,uring_queue.cpp}`：POSIX 后端在批量入队部分失败时执行两阶段回滚（逆序清除未提交槽位 + 循环收割已提交飞行 IO），修复 `uringQueue` 将提交短计数误判为成功及固定计数循环导致的 `EBUSY` 死锁 |

---

## 2. 核心架构机制与数学形式化深挖

### 2.1 TransformerEngine `FusedAttention` 零图断裂自定义算子与确定性 dGLU (`7e7898fa82`, `c25fa27525`)

1. **`te_fused_attn_fwd` / `te_fused_attn_bwd` 扁平签名 (`7e7898fa82`)**：TorchDynamo 无法直接追踪携带复杂 Python 对象（如 `Quantizer`、`AttentionParams`）的 `torch.autograd.Function`。该提交将量化标量（`scale`/`amax`/`scale_inv`）和配置参数拆解为扁平的 `Tensor`/`int`/`float`/`str` 参数列表，注册 `tex.te_fused_attn_fwd` 与 `tex.te_fused_attn_bwd` 及其 `@torch.library.register_fake` 元函数，使 `torch.compile(fullgraph=True)` 在 BF16/FP16 及非 `DelayedScaling` FP8 下实现 **$0$ 图断裂**（仅保存 `18` 个反向传播必需张量）。
2. **dGLU 分组 MLP 确定性 `dprob` 归约 (`c25fa27525`)**：在分组 MoE SwiGLU/GeGLU 反向传播中，概率梯度为：

$$\frac{\partial \mathcal{L}}{\partial p_i} = \sum_{j=1}^{C} g_{i,j}\,\sigma(x_{i,j})\,u_{i,j}$$

旧内核使用跨列块 `atomicAdd` 直接累加至 `dprob[i]`，因此当启用 `torch.use_deterministic_algorithms(True)` 时会抛出运行时异常。新实现分配形状为 $(R, \lceil C / K_{\text{cols}} \rceil)$ 的 FP32 临时工作区（或对 4096 行分块以将峰值显存控制在 $O(4096 \times N_{\text{chunks}})$），先由各列块无竞争写出部分和，再经确定性树归约得到完全可复现的 `dprob`。

### 2.2 Megatron-LM MFSDP Unshard/MXFP8 双事件同步与 TorchTitan EMA 重播 (`17fc7ba109`, `ba53205255`)

- **MFSDP MXFP8 双缓冲独立事件 (`17fc7ba109`)**：当启用 MXFP8 参数双缓冲（`model_weight_buffer` + `colwise_model_weight_buffer`）时，旧代码对两次异步 `all_gather` 复用同一个 `CUDAEvent`，导致首次录制被二次录制覆盖；修复后为每个参数组配置独立的 `(all_gather_event, all_gather_colwise_event)`，并通过单一布尔判据 `model_weight_buffer_all_gather_finished` 决定是否执行等待或即时重分片，并在 `is_Resize_storage_available()` 为假时回退至安全张量切片重绑而非 `storage().resize_(0)`。
- **TorchTitan `EMAStateAdapter` 跨后端重播 (`ba53205255`)**：当从不含 `ema` 键的基础预训练权重恢复、或传入 `--checkpoint.model-only` / `--checkpoint.exclude-from-loading=ema` 时，旧版仅在 DCP 后端的特定分支触发重播，导致 TorchSnapshot 等后端下 EMA 仍停留在随机初始化参数 $\theta_{\text{init}}$；新适配器将 `reseed_from_model()` 统一收口到 `EMAStateAdapter.load_state_dict`，当输入字典为空时直接同步 $\theta_{\text{ema}} \leftarrow \theta_{\text{model}}$。

### 2.3 Mooncake 租户控制面鉴权、TENT 非 Primary CUDA Context 恢复与 NIXL POSIX 回滚 (`edd323921d`, `e3e9d70eea`, `f7533eb8a5`)

- **Mooncake Store-RS 租户绑定 (`edd323921d`)**：在 RPC 连接握手或首包请求时通过 `validate_tenant_binding` 验证调用方租户令牌与目标命名空间绑定，阻断跨租户元数据读写（返回错误码 `-3250`）。
- **Mooncake TENT `CudaContextGuard` 精确恢复 (`e3e9d70eea`)**：旧版 `cudaSetDevice(prev_dev)` 会隐式激活 `prev_dev` 的 **primary context**，破坏调用方正在使用的自定义 `CUcontext`（如驱动 API 创建的专用流上下文）或无上下文线程状态；改为 `cuCtxGetCurrent(&saved_ctx_)` + `cuCtxSetCurrent(saved_ctx_)` 后实现 100% 精确上下文还原。
- **NIXL POSIX 部分入队失败两阶段回滚 (`f7533eb8a5`)**：在 `aio` / `uring` / `linuxaio` 批量提交中，若第 $k$ 个描述符入队失败，且前 $k$ 个描述符中已有 $m$ 个被内核异步消费提交，直接释放整个请求会导致内核写坏已释放内存或触发 `EBUSY` 死锁；修复后先逆序撤销未提交的 $k-m$ 个槽位，再循环调用 `checkCompleted()` 直至 `numCompleted == numSubmitted` ($m=2$)。

---

## 3. 本地确定性复现实验结果 (`own-offline`)

运行 `python experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/run.py --check` 得到的密封结果（[`summary.json`](../../experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/summary.json)）：

| 检验维度 | 旧路径 / 基线表现 | 修复/优化后表现 | 关键量化指标 |
|---|---|---|---|
| TE `FusedAttention` `torch.compile` | `2` 次图断裂 (`FusedAttnFunc`) | `0` 次图断裂 (`te_fused_attn_fwd/bwd`) | `fullgraph=True` 覆盖率 **`85.7%`** (6/7 量化模式) |
| TE dGLU 确定性 `dprob` ($8192\times 4096$) | 抛出确定性异常 (`atomicAdd`) | 2D/3D chunked 显式归约通过 | 4096 行分块节省 **`50.0%`** 峰值工作区显存 |
| Megatron-LM MFSDP MXFP8 Unshard | 单事件被覆盖 (`1` event/group) | 双事件隔离 (`2` events/group) | 兼容无 `resize_storage` 环境与即时重分片 |
| TorchTitan EMA 加载基础模型权重 | 残留随机权重 (`-0.25`) | 自动重播同步模型权重 (`1.75`) | 覆盖所有 Checkpoint 后端与 `--model-only` |
| Mooncake TENT / NIXL POSIX 异常恢复 | 覆盖自定义 `CUcontext` / 卡死 `EBUSY` | 恢复 `ctx_custom_7` / 收割全部 `2` 个飞行 IO | NIXL `uring` 短提交检测 + 零悬空 IO 内存泄漏 |

---

## 4. 解决方案架构师行动指南与可证伪问题

1. **架构师检查清单**：
   - 在开启 `torch.compile` 的 TransformerEngine 训练作业中，优先使用 `FP8CurrentScaling`、`MXFP8BlockScaling` 或 `NVFP4BlockScaling` 以享受 `FusedAttention` 零图断裂；若仍使用 `DelayedScaling`，需知悉其因依赖 Python 侧历史 `amax` 状态机仍会触发 graph break。
   - 在 Kubernetes 滚动升级 Dynamo 推理前端时，确保负载均衡器或 SDK 正确遵从 `503` 响应中的 `Connection: close`，避免长连接持续向排空 Pod 发送请求。
2. **后续可证伪问题**：
   - 当 grouped MLP 的路由 token 数超过 $65536$ 时，dGLU 确定性 4096-row 分块循环与单次 2D 全量工作区归约在 H200/B200 上的内核启动开销交叉点是多少？
