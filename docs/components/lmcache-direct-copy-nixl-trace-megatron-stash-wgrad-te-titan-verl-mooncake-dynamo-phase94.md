# LMCache SGLang 零 SM 批量直拷、NIXL UCX 全链路 Trace 不变量、Megatron-LM Paged-Stash 预热消峰与 GTP CUDA Graph Wgrad 释放、TransformerEngine 设备端 Sentinel、TorchTitan FA 探针、veRL DeepEP/NVSHMEM、Mooncake Master VLOG(1) 10x 提速与 Dynamo 工具名校验解析（Phase 94）

> **English Version**: [lmcache-direct-copy-nixl-trace-megatron-stash-wgrad-te-titan-verl-mooncake-dynamo-phase94.en.md](lmcache-direct-copy-nixl-trace-megatron-stash-wgrad-te-titan-verl-mooncake-dynamo-phase94.en.md)  
> **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`  
> **Experiment Path**: `experiments/lmcache_direct_copy_nixl_trace_megatron_stash_wgrad_te_titan_verl_mooncake_dynamo_phase94/`

---

## 1. 架构背景与核心问题

Phase 94 Track 3 深入覆盖 KV 缓存与传输层（LMCache `#6062`、NIXL `#2467`、Mooncake `#4786`、Dynamo `#16482`）以及分布式训练/RL 框架（Megatron-LM `#8507` & `#8503`、TransformerEngine `#3739` & `#3732`、TorchTitan `#5546`、veRL `#8354`）的十项核心提交：

1. **LMCache SGLang 组件式 KV 布局零 SM `cudaMemcpyBatchAsync` 直拷（`#6062`）**：
   - SGLang 统一多进程（MP）路径注册的是按组件拆分的 4D 视图 `[NB, BS, NH, HS]`（而非 5D `[2, NB, BS, NH, HS]`）；
   - 新增枚举 `EngineKVFormat::NL_X_NB_BS_NH_HS`（`format 18`）并在 `resolve_block_addressing` 中映射其连续块内跨度，使硬件拷贝引擎通过 `cudaMemcpyBatchAsync` 直接在主机锁页内存与各层 K/V 张量间搬运（每 `(kv, layer, block)` 仅约 `0.6 us` CPU 开销，**占用 `0` 个 GPU SM**，彻底绕过 4-slot 临时缓冲与 SM 拷贝 Kernel）。
2. **NIXL UCX 全链路 W3C `traceparent` 与阶段时间戳不变量验证（`#2467`）**：在真实 UCX loopback 传输上验证 `postXferReq.write`、`nixl::submit`、`nixl::wire.submitted`、`nixl::notif.sent` 与 `nixl::notif.received` 五个事件/Span 完整携带请求级 `context=<traceparent>`，且事件微秒时间戳严格满足 `at_us == nixl.phase.timestamp_us`。
3. **Mooncake Master DFS 满载时降级 `VLOG(1)` 解除分片写锁串行化（`#4786`）**：
   - 当 DFS 容量耗尽（且延迟释放窗口内持续数十秒无可用块）时，`MasterService::AllocateReplicas` 原本在持有元数据分片写锁期间对每次失败的 `PutStart` 输出约 `470 B` 的 `LOG(ERROR)`，导致所有并发线程在 `glog` 全局互斥锁上排队；
   - 将 `ErrorCode::NO_AVAILABLE_HANDLE` 改为与内存耗尽一致的 `VLOG(1)`（批次级仍由 `BatchPutStart` 与客户端打单条 `WARNING`），实测单线程失败快路径吞吐从 `0.35 -> 3.5 Mops/s`（**`10.0x` 提速**），16 线程从 `0.26 -> 1.7 Mops/s`（**`6.54x` 提速**）。
4. **Megatron-LM MoE Paged-Stash 预热峰值显存消峰与 GTP CUDA Graph Wgrad 生命周期修复（`#8507`, `#8503`）**：
   - `#8507`：在 MoE Paged-Stash 从 `'capture'` 转入 `'captured'` 并按 `cuda_factor = 1.10` 分配持久化 `stash_buffers` 前，调用 `torch.cuda.synchronize()` 与 `torch.cuda.empty_cache()`（非流捕获态下），释放首轮预热产生的未分片瞬态激活，避免“瞬态预热激活 + 持久化缓冲区”叠加导致的 HBM 峰值冲高（在 `12 GiB` 瞬态激活 + `11 GiB` 持久化缓冲场景下削减 `11.0 GiB` 峰值）；
   - `#8503`：在 Generalized TP（GTP）`_reduce_scatter` 中移除 `torch.distributed._coalescing_manager` 的 `device=wgrads[0].device` 参数，使单分组快路径不再被外层 `_CoalescingManager` `Work` 句柄包裹，确保 `work.wait()` 后立即释放 eager 预热轮的 `wgrad` 输入张量，避免跨 CUDA Graph Replay 泄漏。
5. **TransformerEngine 自定义算子设备端 `None` Sentinel 与 Userbuffers 解耦（`#3739`, `#3732`）**：
   - `#3739`：在 `_register_base_op` 的 `_impl` 与 `_fake` 中使用 `_pack_on_input_device`（`with device:`）包装 `pack_result`，使代表 `None` 的空张量 sentinel 分配在首个非 CPU 输入所在的 CUDA 设备上而非默认 `cpu`，消除 PyTorch Inductor / CUDA Graph 因混合设备输出触发的分区断裂；
   - `#3732`：移除 `CommOverlapHelper`、`CommOverlap`、`CommOverlapP2P` 冗余的 `torch::CustomClassHolder` 基类，同时保留 `NcclCommSharedPtr` 引用计数生命周期。
6. **TorchTitan FlashAttention Wheel 探针、veRL DeepEP 1.2.1 + NVSHMEM 3.4.5 与 Dynamo 带点工具名（`#5546`, `#8354`, `#16482`）**：
   - `#5546`：在 SM100（`FA4 -> flash_attn.cute`）与 SM90（`FA3 -> flash_attn_interface`）上通过 `importlib.util.find_spec` 预检 Wheel 是否安装，避免 Blackwell 静默退化到 Hopper `FA3`，并在 MLA（`d_qk != d_v`）缺失对应包时抛出明确修复指引；
   - `#8354`：在 veRL `uv.cu130.dev5`（Torch 2.13 / CUDA 13）训练镜像与 lockfile 中补齐 `deep-ep==1.2.1` 及 `RUNPATH` 指向的系统级 `NVSHMEM 3.4.5`（`libnvshmem_host.so.3`）；
   - `#16482`：在 Dynamo OpenAI 前端工具校验与 `qwen3_coder`/`harmony` 解析器中放行含 `.` 的函数名（如 `my-tool.v2`）。

---

## 2. 确定性实验验证汇总（`own-offline`）

运行 `python experiments/lmcache_direct_copy_nixl_trace_megatron_stash_wgrad_te_titan_verl_mooncake_dynamo_phase94/run.py --check` 通过全部断言：

| 组件与 PR | 关键指标 | 优化/修复前 | 优化/修复后 | 架构收益 |
|---|---|---:|---:|---|
| LMCache `#6062` SGLang KV 直拷 | `format 18` 占用 SM 数 / 每块 CPU 开销 | 占用 SM + 4-slot 缓冲 | `0` SMs / `0.6 us` | 硬件 Copy Engine 异步直拷 |
| NIXL `#2467` UCX Trace | `traceparent` 与 `at_us` 对齐阶段数 | 仅局部验证 | `5 / 5` 全阶段对齐 | 跨节点 RDMA 精确可观测性 |
| Mooncake `#4786` Master DFS | 满载 `PutStart` 吞吐（1 线程 / 16 线程） | `0.35` / `0.26 Mops/s` | `3.50` / `1.70 Mops/s` | `10.0x` / `6.54x` 消除 glog 锁串行 |
| Megatron-LM `#8507` / `#8503` | Paged-Stash 预热峰值 / GTP Wgrad 驻留 | `23.0 GiB` / 泄漏 | `12.0 GiB` / 立即释放 | 节省 `11.0 GiB` 峰值 + 零跨图驻留 |
| TransformerEngine `#3739` | 自定义算子 `None` sentinel 所在设备 | `cpu` (触发图分裂) | `cuda:0` (零分裂) | 保持完整 CUDA Graph 捕获树 |
| TorchTitan `#5546` / veRL `#8354` | SM90/SM100 FA 探针 / DeepEP+NVSHMEM | 运行时崩溃 / 缺失库 | 预检回退 / `1.2.1 + 3.4.5` | 健壮算子路由与 MoE EP 闭环 |
