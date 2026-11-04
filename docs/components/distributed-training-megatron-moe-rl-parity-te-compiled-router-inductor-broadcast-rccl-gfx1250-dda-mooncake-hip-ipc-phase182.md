# 分布式训练、集合通信与 KV 传输网络：Megatron-LM FlashInfer MXFP8/NVFP4 RL 训推逐比特对齐、TE 编译态 MoE 路由与零拷贝打包 QKV 反向、PyTorch Inductor 宽广播实例化、RCCL `gfx1250` 批量轮询 DDA LL 与 Mooncake 多协议显存生命周期（Phase 182）

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-moe-rl-parity-te-compiled-router-inductor-broadcast-rccl-gfx1250-dda-mooncake-hip-ipc-phase182.md)** | **[English](distributed-training-megatron-moe-rl-parity-te-compiled-router-inductor-broadcast-rccl-gfx1250-dda-mooncake-hip-ipc-phase182.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 182（`priority: 191`）审计了 `NVIDIA/Megatron-LM`、`NVIDIA/TransformerEngine`、`pytorch/pytorch`、`ROCm/rocm-systems`（RCCL）、`kvcache-ai/Mooncake`、`LMCache/LMCache` 与 `ai-dynamo/nixl` 的 **14 个上游主干提交**：

1. **Megatron-LM FlashInfer MXFP8 / 每 Token NVFP4 RL 训推逐比特对齐、融合批量不变平方 ReLU + MXFP8 量化与 GRPO 观察器保留**：
   - `ebcf9d67221d` (#7798) 新增面向强化学习（RL）训练的可选开关 `moe_training_matches_inference: true`：在启用 `inference_optimized`、`batch_invariant_mode`、`moe_token_dispatcher_type='alltoall'` 与选择性 `moe` 重计算时，训练的 value pass 与 `eval()` log-prob 前向直接调用与推理生成完全相同的 FlashInfer 路由 MXFP8 或每 Token NVFP4 内核（配合 `unpermute_with_bf16_rank_partials` 按生成顺序执行 FP32 FMA 并量化为 BF16），而反向传播保持 Transformer Engine（TE）重计算梯度不变（相对 BF16 距离满足 $d_{\mathrm{FI}} \le 1.004 \cdot d_{\mathrm{TE}}$）。在 4x GB300（`EP=4`，Nano-v3 128 专家 top-6）上：
     - **每 Token NVFP4**：实现训练与推理生成 log-prob **逐比特完全相等（`0` ULP 偏差）**；log-prob 前向较 TE NVFP4 提速 **`3.37x–4.357x`**（`4,096` tok/rank 从 `23.53 ms` 降至 **`5.40 ms`**，消除了 TE 逐专家 Python 循环的 ~700 次内核启动）；训练整步较 TE NVFP4 + 重计算提速 **`2.28x–2.897x`**（`1,024` tok/rank 从 `59.47 ms` 降至 **`20.53 ms`**，较 BF16 TE + 重计算亦快 `1.05x–1.21x`），且每层每 Rank 节省 **`178 MiB`** 静态显存（`1,543 vs 1,721 MiB`）。
     - **MXFP8**：实现逐比特 log-prob 对齐，log-prob 前向提速 **`1.40x–1.422x`**（`4,096` tok/rank 从 `9.27 ms` 降至 **`6.52 ms`**），训练整步提速 **`1.13x–1.151x`**（`1,024` tok/rank 从 `31.43 ms` 降至 **`27.30 ms`**）。
   - `4a4064f0631c` (#8645) 将批量不变模式下的 `squared_relu_with_probs` 与 `mxfp8_quantize_rows` 融合为单内核 `squared_relu_with_probs_quantize_mxfp8`，在 GB200 `Nemotron-3-Nano` 上每解码步节省 `0.07–0.13 ms`；`993bf83ec875` (#8850) 修复 GRPO 空 `create_cudagraphs()` 调用提前移除 saved-tensor observer 的缺陷。
2. **TransformerEngine 编译态融合 MoE 路由与零拷贝打包 QKV 反向、PyTorch Inductor 宽广播生产者实例化与 AOTI C-Shim CUDA Graph**：
   - `720b25258be3` (#3789) 在 TE 默认自注意力 eager 路径中用 `SplitAlongDim` 替代 `Tensor.select` 切分打包的 `qkv_layer`/`kv_layer`，消除反向传播中为每个视图构造全尺寸零填充梯度张量并求和的额外显存分配与拷贝（**`0` 次零填充视图梯度分配**）；`2d65f35b662c` (#3691) 通过 `setup_context` 与可变自定义算子为融合 MoE 路由与分位数负载均衡（QB）直方图提供 `torch.compile(fullgraph=True)` 支持（`3,391` 项路由测试通过）。
   - `1933a0a9ee51` (#202772) 在 PyTorch Inductor 调度器中对宽度 $\ge 64$（`realize_broadcast_reuse_threshold=64`）的广播消费者启用读取次数启发式（`realize_reads_threshold`），防止按行计算的昂贵生产者被内联至宽消费者的每个元素中重复计算，将 B200 上 `gpt-oss` MoE 下投影后处理（epilogue）耗时从 `36.4 us` 降至 **`16.8 us`（`2.167x` 加速，`-53.85%`）**。
   - `b31d68db9561` (#202227) 在 PyTorch AOTI C shim 中暴露 `at::cuda::MemPool`、`at::cuda::CUDAGraph` 与 `aoti_torch__foreach_copy_`，支持模型 `.so` 在线程局部模式下捕获 CUDA Graph 并拦截 `cudaMallocAsync` 后端冲突。
3. **ROCm RCCL `gfx1250` 批量轮询预取 DDA LL AllReduce 与 ReduceScatter 分层上限调优**：
   - `f59fd15cb79a` (#14880) 在 `gfx1250`（MI455X）DDA LL one-shot AllReduce 中于 epoch 屏障前将首批发送包预取至 L2（`ddaLLPrefetchSend`），并在阶段 2 将对等行轮询按最多 **`8` 行一批**合并至单次加载等待（`ddaLLWaitLineFlagsBatched`），消除逐行串行往返等待。
   - `3b0c0acddf20` (#14759) 调优 `gfx1250` SPX ReduceScatter 阈值：将 DDA LL128 上限提高至 `< 8` Ranks 时总计 **`224 MiB`**、`>= 8` Ranks 时每 Rank **`7.5 MiB`**（受 `256 MiB` 共享上限约束），并将 DDA LL 上限提高至 `< 8` Ranks 时总计 `4 MiB`、`>= 8` Ranks 时每 Rank **`1.0 MiB`**，避免在 4–64 卡 `32M–256M` 区间过早回退至较慢的 Ring 内核。
4. **Mooncake 多协议显存所有权与 Redis 自动重连、LMCache `RawBlock` 队列定容与 NIXL 零分配 NVTX / EFA 门控**：
   - `748af1a53dba` (#3548) 在 Mooncake TransferEngine 中按 `hipMemGetAddressRange` 分配基址粒度注册 HIP IPC 显存，建立 `addr -> base` 引用计数映射与按协议标签（`"hip"` / `"rdma"` / `"tcp"` 共 **`3` 种传输协议**）过滤的 `removeLocalMemoryBuffer` 谓词，彻底根除 8x MI308X / MI355X 上缓存分配器子张量重定位导致的数据错位及 `rdma+hip` 基址描述符互删泄漏；`4704c67f13ff` (#4042) 为 `RedisStoragePlugin` 增加 **`3 s`** 连接/命令超时、TCP Keepalive 以及连接断开后自动重连并重放 `AUTH`/`SELECT` 的自愈机制（解决负载均衡器 ~4000s 空闲回收导致进程永久失联）。
   - `338d5fba394e` (#4697) 在 LMCache `RawBlock` 中为工作线程 `in-flight` 表中的每个未完成请求预留完成队列（CQ）容量直至收割完成，将越界队列溢出崩溃降为 **`0`**；`e2cd0283d5df` (#2561) 在 NIXL 中仅当检测到 EFA 设备时加载 UCX EFA 插件（消除无 EFA 容器中的退出码 `139 SIGSEGV`）；`13c52d21bdc6` (#2571) 通过 **64** 槽位编译期 FNV-1a 哈希表与 64B 栈上缓冲区解析 NVTX 事件名，在 `nixlbench` 中每轮迭代减少 **`6` 次**堆分配（全采样从 `92` 降至 `86`，未采样从 `42` 降至 `38`）。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+--------------------------------------------------------------------------------------------------------+
| Phase 182 Distributed Training, Collectives & KV Fabric: Megatron-LM, TE, PyTorch, RCCL & Mooncake/NIXL|
+--------------------------------------------------------------------------------------------------------+
  [Megatron-LM RL Train-Gen Bitwise Parity & Fused Batch-Invariant Quant (#7798, #8645, #8850)]
    |-- FlashInfer RL Value/LogProb (#7798)--> Bitwise Parity (0 ULPs): NVFP4 LogProb 4.357x, Train 2.897x (-178MiB)
    |-- Fused SqReLU+MXFP8 Quant (#8645)   --> squared_relu_with_probs_quantize_mxfp8 (-0.07..0.13 ms/step GB200)
    +-- GRPO Saved-Tensor Observer (#8850) --> Preserve Observer Across Empty create_cudagraphs() in Old LogProb
                                           |
                                           v
  [TransformerEngine, PyTorch Inductor/AOTI & ROCm RCCL gfx1250 (#3789, #3691, #202772, #202227, #14880, #14759)]
    |-- TE SplitAlongDim & Router (#3789)  --> 0 Zero-Filled Packed-QKV Grad Views + Fullgraph Compiled MoE Router
    |-- PyTorch Inductor & AOTI (#202772)  --> Wide-Broadcast (>=64) Producer Realize (2.167x B200) + C-Shim Graph
    +-- RCCL gfx1250 DDA LL/LL128 (#14880) --> L2 Send Prefetch + 8-Line Batched Poll + RS 224MiB/7.5MiB Tier Caps
                                           |
                                           v
  [Mooncake HIP IPC / Redis Recovery, LMCache RawBlock & NIXL UCX/NVTX (#3548, #4042, #4697, #2561, #2571)]
    |-- Mooncake Multi-Transport (#3548)   --> Base-Granular HIP IPC + 3-Protocol Ownership + Redis 3s Reconnect
    +-- LMCache & NIXL Hardening (#4697)   --> In-Flight CQ Reservation + EFA Device Gate + 64-Slot FNV-1a NVTX
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/megatron_moe_rl_parity_rccl_gfx1250_dda_mooncake_hip_ipc_phase182/run.py`](../../experiments/megatron_moe_rl_parity_rccl_gfx1250_dda_mooncake_hip_ipc_phase182/run.py) `--check` 100% 验证通过：
- `megatron_nvfp4_logprob_max_speedup = 4.357`、`megatron_nvfp4_train_max_speedup = 2.897`、`megatron_nvfp4_static_mem_saved_mib = 178`
- `megatron_mxfp8_logprob_max_speedup = 1.422`、`megatron_train_gen_bitwise_logprob_mismatch_ulps = 0`
- `pytorch_inductor_wide_broadcast_speedup = 2.167`（`36.4 -> 16.8 us`）、`te_packed_qkv_zero_fill_grad_allocations = 0`
- `rccl_gfx1250_dda_ll_batched_poll_lines = 8`、`rccl_gfx1250_rs_ll128_sub8rank_cap_mib = 224.0`、`rccl_gfx1250_rs_ll128_per_rank_cap_mib = 7.5`
- `mooncake_transport_ownership_protocols = 3`、`mooncake_redis_reconnect_timeout_s = 3`、`lmcache_rawblock_cq_overflow_panics = 0`、`nixl_nvtx_allocs_saved_per_iter = 6`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **Blackwell 大规模 MoE 强化学习（RL / GRPO）后训练** | 在 GB200/GB300 上运行 FlashInfer 路由 NVFP4 或 MXFP8 rollout 的 RL 训练时启用 `moe_training_matches_inference: true`（`#7798`），消除训练与生成之间的重要性采样比率偏差并获得 `4.357x` NVFP4 log-prob 加速与 `2.897x` 训练步加速。 | 在训练与生成使用不同 EP 维度、`ETP > 1`、或使用 DeepEP/HybridEP flex 调度器时启用该开关（其合并归约非 EP 不变，会被配置校验显式拒绝）。 |
| **MI455X (`gfx1250`) 集合通信与 PyTorch/TE 编译加速** | 在 4–64 卡 MI455X 上使用调优后的 RCCL ReduceScatter DDA LL（`1 MiB/rank`）与 LL128（`7.5 MiB/rank`）上限及 8 行批量轮询 AllReduce（`#14759/#14880`），并在 B200 上启用 Inductor 宽广播实例化（`#202772`，`2.167x`）。 | 在启用 `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` 的进程中使用 AOTI C-shim 创建独立 `MemPool` 捕获多流并发 CUDA Graph（`#202227` 显式拦截该冲突）。 |
| **MI308X / MI355X 跨进程 KV 传输与长周期 Redis 元数据服务** | 在 Mooncake `rdma+hip` 多协议环境中升级至基址粒度引用计数与按协议标签隔离的 BufferDesc 管理（`#3548`），并为云端 TCP 负载均衡器后的 Redis 元数据插件启用 3s 超时与自动重连（`#4042`）。 | 在 LMCache `RawBlock` 异步落盘时仅凭提交队列（SQ）剩余槽位准入新批次而忽略未收割的完成队列（CQ）in-flight 请求（`#4697`）。 |
