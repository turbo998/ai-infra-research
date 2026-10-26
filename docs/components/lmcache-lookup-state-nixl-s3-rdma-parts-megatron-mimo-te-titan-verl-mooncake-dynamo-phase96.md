# LMCache 回调式 LookupState、NIXL 超 4GiB 分片 S3-over-RDMA、Mooncake 命名空间驱逐框架、verl 采样器概率校验与 Dynamo 共享 ZMQ Context（Phase 96）

> **证据分类**：`implementation`（固定上游源码提交）+ `own-offline`（本项目纯 Python 3.12 标准库离线确定性验证，非 GPU/RDMA 硬件实测 profiling）。

## 1. 核心上游源码实证（10 项零重叠一手提交）

| Source ID | 仓库 / 提交 | 关键实现路径与架构机制 |
|---|---|---|
| `lmcache-mp-callback-driven-lookup-state-scheduler-pr5960-2026` | `LMCache/LMCache` `ff9185d419` (PR #5960) | 引入 `LookupState`（`lmcache/integration/vllm/mp_lookup_state.py`），通过 `MessagingFuture` 完成回调串联 `LOOKUP -> QUERY_PREFETCH_STATUS` 并自动释放多服务端未取回的尾部读锁（tail locks）。 |
| `lmcache-core-l1-l2-allowed-list-prefetch-storage-manager-pr5922-2026` | `LMCache/LMCache` `a81b4e5aea` (PR #5922) | 在向存储管理器提交预取请求（`controller.submit_prefetch_request(spec)`）时支持 L1/L2 允许列表（allowed list）过滤。 |
| `nixl-obj-move-rdma-objects-larger-cuobject-in-parts-pr2406-2026` | `ai-dynamo/nixl` `99d469be5d` (PR #2406) | S3-over-RDMA 插件将超过单次 `cuObject` 注册上限 `CUOBJ_MAX_MEMORY_REG_SIZE`（$4\text{ GiB} - 64\text{ KiB} = 4,294,901,760\text{ B}$）的 `rdmaPutObject`/`rdmaGetObject` 拆分为多个有界分片传输，消除大显存池下 `CU_OBJ_INVALID_VALUE` 报错。 |
| `nixl-core-preserve-active-request-state-release-reject-pr2175-2026` | `ai-dynamo/nixl` `5b565cb9a6` (PR #2175) | `nixlAgent::releaseXferReq` 在 `req_hndl->status == NIXL_IN_PROG` 时使用局部变量 `release_status` 接收 `engine->releaseReqH` 返回值；若后端拒绝释放（`release_status < 0`），保持 `req_hndl->status = NIXL_IN_PROG` 可继续轮询。 |
| `megatron-honor-rope-args-nemotron-mimo-provider-pr8371-2026` | `NVIDIA/Megatron-LM` `964b1deefb` (PR #8371) | Nemotron MIMO 语言模型 provider 改为透传 CLI 配置的 `position_embedding_type`、`rotary_base`、`rotary_percent` 与 `seq_len_interpolation_factor`，修复原先硬编码 `"none"` 覆盖 RoPE 的缺陷。 |
| `te-jax-fix-undefined-sr-rng-state-dry-run-encoder-pr3498-2026` | `NVIDIA/TransformerEngine` `f3826223b7` (PR #3498) | 修复 JAX 单卡多进程与模型并行 encoder 示例在 `--dry-run` 下缺失 `SR_KEY: sr_rng` 导致的未定义随机状态报错。 |
| `torchtitan-use-torch-accelerator-device-stream-memory-pr5450-2026` | `pytorch/torchtitan` `35f4a2e166` (PR #5450) | 将 EMA CPU 卸载、RL 权重暂存、优化器首步同步与 `graph_trainer` 预编译中的 `torch.cuda` 全部统一至 `torch.accelerator`、`torch.Stream` 与 `torch.Event`。 |
| `verl-validate-sampler-prob-records-before-support-align-8eac5b0-2026` | `volcengine/verl` `8eac5b0170` | 在 `extract_response_topk_logprobs` 执行支撑集对齐前，强制校验候选 `head_ids` 无重复、$\sum \exp(\text{head\_log\_probs}) \le 1.0 + 10^{-5}$，并核对被采样 token 的 logprob 一致性。 |
| `mooncake-store-eviction-framework-over-namespace-layer-pr4811-2026` | `kvcache-ai/Mooncake` `fff8e90952` (PR #4811) | 在命名空间层之上构建 `Evictor`、`EvictionPolicy`（硬固定/活动租约保护 + 软固定两阶段延迟驱逐）与 `OffloadingReclaimer`（先落盘再释放内存副本的 `kDemoted` 路径）。 |
| `dynamo-kv-router-share-zmq-contexts-dial-replay-per-gap-pr17253-2026` | `ai-dynamo/dynamo` `8222ff098c` (PR #17253) | KV-Router 跨所有事件订阅 socket 共享进程级 `OnceLock<zmq::Context>`（默认 `DYN_ZMQ_IO_THREADS=4`），并在检测到序号缺口时按缺口按需拨号 DEALER replay socket。 |

## 2. 数学建模与架构师计算练习

### 2.1 NIXL S3-over-RDMA 超大对象分片公式
由于 `cuMemObjGetRDMAToken` 单次内存注册上限为：
$$L_{\max} = 4\text{ GiB} - 64\text{ KiB} = 4,294,901,760\text{ B}$$
对于 $S = 10\text{ GiB} = 10,737,418,240\text{ B}$ 的大模型 KV/权重对象，分片数为：
$$N_{\text{parts}} = \left\lceil \frac{S}{L_{\max}} \right\rceil = 3$$
各分片字节数分别为 `[4294901760, 4294901760, 2147614720]`，总和严格等于 $10\text{ GiB}$。

### 2.2 Dynamo KV-Router 共享 ZMQ Context 线程收敛与 LMCache 尾锁释放
- 当路由节点订阅 $32$ 个 DP worker 的 KV 事件流且每个 Context 配置 `4` 个 I/O 线程时，共享单 `OnceLock<zmq::Context>` 将 ZMQ 后台线程数从 $32 \times 4 = 128$ 降至 $4$（**收敛比 $32.0\times$**）。
- 在 LMCache 多服务端查找中，若两个分片服务端返回命中长度 `[384, 512]`，对齐公共前缀为 `384` tokens，回调状态机自动向第 2 个服务端发送 `[384, 512)` 区间的 `128` tokens 尾锁释放请求（`leaked_tail_locks = 0`）。

## 3. 可复现离线实验与验证

运行命令：
```bash
python experiments/lmcache_lookup_state_nixl_s3_rdma_parts_megatron_mimo_te_titan_verl_mooncake_dynamo_phase96/run.py --check
```
实验产物 `summary.json` 已验证：
- `released_tail_lock_tokens == 128`，`leaked_tail_locks == 0`，`num_rdma_parts == 3`，`status_after_rejected_release == "NIXL_IN_PROG"`；
- `reclaimed_bytes == 512`，`demoted_keys == ["t1/k1"]`，`offloaded_then_demoted_keys == ["t1/k2"]`，`thread_reduction_ratio == 32.0`；
- `valid_head_accepted == True`，`duplicate_head_rejected == True`，`impossible_mass_rejected == True`（`1.72357 > 1.00001`）。
