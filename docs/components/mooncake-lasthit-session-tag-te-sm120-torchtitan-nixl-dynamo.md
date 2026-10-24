# Mooncake Store `LastHitOnly` 批量探测租约与会话归属标签、TransformerEngine `SM120` Gated FP8 共享内存护栏、TorchTitan RL YaRN 注意力缩放传递与 LMCache/NIXL/Dynamo 混合 KV 路由深度解析

> **语言 / Language**: [中文](mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo.md) | [English](mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo.en.md)  
> **Phase**: 71 (`priority: 80`, Track C — Distributed Training, KV Storage & RDMA Collectives)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `author-benchmark` (作者 TorchTitan GPT-OSS-20B RL KL 散度实测 `0.055 -> 0.0006`) + `own-offline` (本项目确定性标准库仿真 `experiments/mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo/run.py`)

---

## 1. 架构背景与核心工程问题

在分布式 KV 缓存引擎（Mooncake、LMCache、Dynamo）、RDMA/DOCA 传输运行时（NIXL）与分布式训练/RL 框架（TransformerEngine、TorchTitan、Megatron-LM）的最新演进中，暴露出四个核心系统工程问题：

1. **Mooncake Store `BatchProbeKey` 在长前缀探测下的过度读租约（Over-Leasing）与并发驱逐回退 (`kvcache-ai/Mooncake@5d77d3446b9f`, `#4314`) 及跨副本写入会话归属追踪 (`@5b2069a120b2`, `#4385`)**：当推理引擎按位置顺序探测 $N$ 个候选前缀块（每个候选包含 `candidate_size` 个跨层/跨 TP Rank 键）时，若对所有命中的前缀块都授予读租约（Read Lease），会无谓锁定大量不会被实际读取的中间块并阻碍正常 LRU 驱逐；若完全不授租约，选中的最长前缀又可能在随后 `BatchGet` 前被驱逐。如何通过 `ProbeLeaseMode::LastHitOnly` 仅对最后一个完整命中的候选组加锁重验并授予读租约？
2. **TransformerEngine `cast_fp8_gated_kernel` 在 `SM120` 上的 `99 KiB` 共享内存越界崩溃 (`NVIDIA/TransformerEngine@62b13895c2e6`, `#3600`) 与单组 `GroupedMLP` 冻结权重 CuTeDSL `DenseGEMM` FC2 调度 (`@0c4d9610ac3b`, `#3666`)**：为什么在 Blackwell Datacenter (`SM100`, 单 Block 最大 Opt-in SMEM `227 KiB`) 上正常运行的 TMA Gated FP8 转型内核，在 `SM120`（最大 Opt-in SMEM 仅 **`99 KiB`**）上运行 BF16 backward (`131,296 B`) 或 FP32 forward/backward (`164,064 B` / `229,600 B`) 时会直接触发 `cudaErrorInvalidValue`？
3. **TorchTitan RL vLLM 推理包装器丢失 YaRN `softmax_scale` 导致策略-采样 KL 散度激增 91.7 倍 (`pytorch/torchtitan@8a3f5bcaa4fd`, `#5347`) 与统一 `torch.accelerator` 流/事件抽象 (`@35f4a2e16658`, `#5450`)**：为什么在 TorchTitan RL 训练 `GPT-OSS-20B` 或 `DeepSeek-V3` 时，`_replace_vllm_layer_configs` 替换出的 vLLM 注意力层会退回默认 $1/\sqrt{d_k}$ 缩放而丢失 YaRN `mscale` 修正，导致策略模型与 Rollout 引擎之间的 KL 散度高达 `0.055`（修复后降至 **`0.0006`**）？
4. **LMCache P2P 查找限制本地 DRAM L1 (`LMCache/LMCache@c90e6862abb0`, `#6014`)、NIXL 活跃传输释放拒绝后的 `NIXL_IN_PROG` 状态保留 (`ai-dynamo/nixl@5b565cb9a6d4`, `#2175`) 与 GPUNetIO DOCA 回滚 (`@42f1714f4605`, `#2052`)、以及 Dynamo 混合 Mamba/Attention 模型的 KVCR 几何误隔离修复 (`ai-dynamo/dynamo@8bf042187a8f`, `#15535`)**：为什么混合架构模型（如包含 Mamba 循环状态与主注意力的模型）发射的 `KVCR` 缓存驻留事件会导致 Dynamo KV Router 误报 `Incompatible KVCR block geometry; withdrawing cache owner` 并永久撤回整个 Worker 的缓存所有权？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 Mooncake Store `LastHitOnly` 逆序候选重验租约 (`#4314`) 与会话写入标签 (`#4385`)

1. **`MasterService::BatchProbeKey` 的 `ProbeLeaseMode::LastHitOnly` 两阶段协议 (`kvcache-ai/Mooncake@5d77d3446b9f`, `#4314`)**：
   - **参数契约**：在 `LastHitOnly` 模式下，要求 `candidate_size > 0` 且 `keys.size() % candidate_size == 0`（否则每个键返回 `ErrorCode::INVALID_PARAMS`）。调用方将每个候选位置所需的全部层/Rank 分片打包为连续的 `candidate_size` 个键。
   - **阶段 1（无锁/无租约全量探测）**：Master 对全部 `keys` 调用 `ProbeSingleKey(key, /*grant_lease=*/false)`，获取初始存在性位图 `results[0..N-1]`，不延长任何对象的读租约 TTL。
   - **阶段 2（逆序完整候选重验与读租约授予）**：从末尾向头部以步长 `candidate_size` 逆序扫描每个候选区间 `[begin, end)`：
     - 若初始探测中该区间并非全 `true`，直接跳过（零额外加锁开销）；
     - 若初始探测全 `true`，对该区间内的每个键再次获取分片锁并调用 `ProbeSingleKey(key, /*grant_lease=*/true)`，同时用重验结果覆写 `results[i]`；
     - **并发驱逐安全回退**：若在阶段 1 与阶段 2 之间，候选 $p_2$ 的某个分片键恰好被并发线程驱逐，重验会将该位置更新为 `false`（使 $p_2$ 不再呈现伪全 `1`），循环继续向更早的完整候选 $p_0$ 回退并成功授予读租约；一旦某个候选重验全 `true`，立即 `break` 停止扫描。
2. **跨副本会话归属标签 `session_ids` (`@5b2069a120b2`, `#4385`)**：
   - 在 `ReplicateConfig` 与 `Metadata` 中引入 `struct_pack::compatible<std::vector<std::string>, 1> session_ids`（校验非空、UTF-8 合法且 $\le 256$ B）。
   - `PutStart` 调用 `BeginSessionTag(session_id, eligible_replica_ids)` 暂存 `PendingSessionTag`；仅当 `PutEnd` 成功落盘在某个 `eligible_replica_ids` 副本上时才调用 `CommitSessionTag` 将其转为已提交标签，并随 HA 快照持久化及 ZMQ MsgPack `stored` 事件向外广播；若所有合法副本均被 `PutRevoke` 撤销，则通过 `ClearSessionTagIfNoViableReplica` 自动清除标签。

### 2.2 TransformerEngine `SM120` Gated FP8 共享内存预算护栏 (`#3600`) 与单组 CuTeDSL FC2 调度 (`#3666`)

1. **`cast_fp8_gated_kernel` 动态 + 静态共享内存精确拟合 (`NVIDIA/TransformerEngine@62b13895c2e6`, `#3600`)**：
   - 在 `transformer_engine/common/cast/gated/gated_fp8.cuh` 中，TMA Gated Cast 内核使用双缓冲（`BUFFERS_NUM = 2`）、Tile 尺寸 $64 \times 128$（`16,384` 元素）以及 `TMA_SHMEM_ALIGNMENT = 128 B`。动态共享内存计算公式为：
     $$
     S_{\text{dyn}} = S_{\text{grad}} + 2 S_{\text{in}} + 2 S_{\text{out}} + 128\text{ B}
     $$
     此外，内核还包含 `96 B` 的静态共享内存（`32 B` mbarrier 数组 + `64 B` `reduce_max` 归约暂存区），通过 `cudaFuncGetAttributes(&attr, kernel)` 读取 `attr.sharedSizeBytes`。
   - **SM100 (`227 KiB`) vs SM120 (`99 KiB` = `101,376 B`) 精确容量对比**：
     - **BF16 Forward (`in=2B, out=1B`)**：$S_{\text{dyn}} = 0 + 2(32768) + 2(16384) + 128 = 98,432\text{ B}$，加上静态 `96 B` 共 **`98,528 B`** $\le 101,376\text{ B}$（**在 SM120 上刚好可运行 TMA**）；
     - **BF16 Backward (`grad=2B, in=2B, out=1B`)**：$S_{\text{total}} = 131,200 + 96 =$ **`131,296 B`** $> 101,376\text{ B}$（在 SM100 可运行，在 SM120 超限 **回退非 TMA 内核**）；
     - **FP32 Forward / Backward (`in=4B`)**：$S_{\text{total}}$ 分别为 **`164,064 B`** 与 **`229,600 B`**（均超出 SM120 的 `99 KiB` 上限，其中 FP32 Backward 甚至接近 SM100 的 `227 KiB = 232,448 B` 极限）。
2. **单组 `GroupedMLP` 推理/冻结权重启用 CuTeDSL `DenseGEMM` FC2 (`@0c4d9610ac3b`, `#3666`)**：
   - 在 `grouped_mlp.py` 中，当 `num_groups == 1` 且 `not weight_requires_grad`（如 `torch.no_grad()` 或冻结专家权重的推理/微调）时，无需等待 backward 所需的 `columnwise_data` 即可直接派发 `_single_group_fc2_gemm` CuTeDSL DenseGEMM；同时将 `grouped_gemm_dactivation_is_deterministic` 提升至基类动态检测 cuDNN FE 1.28.0+ 的 `deterministic` 参数 (`#3676`)。

### 2.3 TorchTitan RL vLLM 注意力 YaRN `softmax_scale` 传递 (`#5347`) 与 Megatron-LM DSA Spec 派生 (`#8414`)

1. **TorchTitan RL `softmax_scale` 属性暴露与注入 (`pytorch/torchtitan@8a3f5bcaa4fd`, `#5347`)**：
   - 在 `torchtitan/models/gpt_oss/model.py` 与 `deepseek_v3/attention.py` 中，将 YaRN 调整后的注意力缩放因子定义为 `Attention.Config.softmax_scale` 属性：
     $$
     \text{softmax\_scale} = \frac{1}{\sqrt{d_{qk}}} \times \left(0.1 \times \text{mscale} \times \ln(\text{rope\_factor}) + 1.0\right)^2
     $$
   - 对 **GPT-OSS** ($d_{qk}=64, \text{rope\_factor}=32.0, \text{mscale}=1.0$)：未修正值为 $1/\sqrt{64} = 0.125$，YaRN 修正后为 **`0.2266575543`**；
   - 对 **DeepSeek-V3** ($d_{qk}=128+64=192, \text{rope\_factor}=40.0, \text{mscale}=0.7$)：未修正值为 $1/\sqrt{192} \approx 0.07216878$，YaRN 修正后为 **`0.1142519529`**；
   - `_replace_vllm_layer_configs` 将 `attention_config.softmax_scale` 传入 `VLLMAttention.Config(scale=...)`，使 GPT-OSS-20B RL 训练与 vLLM 采样之间的 KL 散度从 `0.055` 骤降至 **`0.0006`（降低 `91.7x`）**。
2. **Megatron-LM 实验性 DSA `dataclasses.replace` 派生 (`NVIDIA/Megatron-LM@a747bac258df`, `#8414`)**：
   - 修复 `get_gpt_layer_with_transformer_engine_submodules` 中开启 `--experimental-attention-variant dsa` 时原地篡改全局缓存的 `ModuleSpec` 的副作用，改用 `dataclasses.replace` 安全克隆。

### 2.4 LMCache 本地 DRAM P2P 限定 (`#6014`)、NIXL 传输状态机护栏 (`#2175`, `#2052`) 与 Dynamo 混合 KVCR 过滤 (`#15535`)

1. **LMCache P2P Lookup 限定 `(DRAM, LOCAL)` L1 (`LMCache/LMCache@c90e6862abb0`, `#6014`)**：`p2p_lookup_and_lock` 在构造 `PrefetchTaskSpec` 时显式传入 `allowed_l1_memory_tags=frozenset({MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)})` 与 `allowed_l2_memory_tags=frozenset()`，确保对等节点查询仅锁定已注册在 RDMA 传输通道上的本地 DRAM L1 块，绝不触发跨盘/远端 L2 异步拉取。
2. **NIXL `releaseXferReq` 活跃传输状态保持 (`ai-dynamo/nixl@5b565cb9a6d4`, `#2175`) 与 GPUNetIO OOB IPv4 早期校验 (`@42f1714f4605`, `#2052`)**：
   - 当调用方对仍在飞行中的传输请求误调 `releaseXferReq` 且后端返回拒绝（如 `NIXL_ERR_REPOST_ACTIVE`）时，不再将 `req->status` 覆写为 `NIXL_ERR_BACKEND`，而是保持 `NIXL_IN_PROG`，使后续 `getXferStatus` 能继续进入 `checkXfer` 完成轮询；
   - 在 GPUNetIO 插件初始化中，将 `oob_connection_get_ip` 提前至 GPU 显存与流分配之前，若网卡无 IPv4 地址立即返回 `-1`，并按 `doca_dev_close -> doca_verbs_pd_destroy -> doca_verbs_context_destroy` 严格逆序回滚 DOCA 资源。
3. **Dynamo KV Router 混合模型 `NonMainAttentionGroup` 过滤前置 (`ai-dynamo/dynamo@8bf042187a8f`, `#15535`)**：
   - 在 `publisher.rs` 的 KVCR 批处理循环中，旧代码先检查 `*block_size != kv_block_size` 再调用 `preprocess_residency_with_reason`。在 Hybrid Mamba/Attention 模型中，Mamba 状态块（`group_idx=1, block_size=64`）与主注意力块（`group_idx=0, block_size=16`）尺寸不同，旧顺序会误触发 `cache_owner_fault` 并撤回整个 Worker 的 CacheOwner！修复后先执行 `preprocess_residency_with_reason` 过滤掉 `NonMainAttentionKind` / `NonMainAttentionGroup`，仅对通过过滤的主注意力组校验 `block_size`。

---

## 3. 本项目确定性离线实验与验证 (`own-offline`)

实验脚本：`experiments/mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo/run.py`（结果固化于 `summary.json`，由 `tests/test_phase71_experiments.py` 校验）：

| 验证子项 | 关键输入 / 参数 | 实测 / 仿真输出 | 架构结论 |
|---|---|---|---|
| Mooncake `LastHitOnly` & 并发回退 | 4 候选 (`size=2`)，`[T,T, F,F, T,T, T,F]`；模拟 $p_2$ 并发失效 | 正常选中 `cand=2`（仅重验 `[2]`）；$p_2$ 失效时自动回退选中 `cand=0` | 避免前缀全量加租约，且在并发驱逐下安全降级至次长完整前缀 |
| TE `cast_fp8_gated_kernel` SMEM 拟合 | `SM100` (`227 KiB`) vs `SM120` (`99 KiB`), `static=96B` | `fwd_bf16=98,528 B` (SM120 通过)；`bwd_bf16=131,296 B` / `fp32=164,064 B` (SM120 拦截回退) | 计入 `96 B` 静态 SMEM 后精确拦截 `SM120` 超限 TMA 内核 |
| TorchTitan RL YaRN `softmax_scale` | GPT-OSS ($d=64, f=32$) / DeepSeek-V3 ($d=192, f=40, m=0.7$) | `0.2266575543` / `0.1142519529`；KL 从 `0.055` 降至 `0.0006` (`91.67x`) | 消除训练与 vLLM Rollout 间的注意力温度失配 |
| LMCache / NIXL / Dynamo 护栏 | Hybrid KVCR `(attn=16, mamba=64)`；NIXL 活跃释放拒绝 | Dynamo 零误故障；NIXL 状态保持 `NIXL_IN_PROG`；LMCache 仅锁 `(DRAM, LOCAL)` | 修复混合模型 CacheOwner 误撤回与 NIXL 状态机死锁 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **计算练习**：在 TransformerEngine `cast_fp8_gated_kernel` 中，若将输入激活改为 FP8 (`1 B`)、梯度为 BF16 (`2 B`)、输出为 FP8 (`1 B`)，在双缓冲 $64 \times 128$ Tile 与 `96 B` 静态共享内存下，计算 Backward 内核的总共享内存需求，并判断其能否在 `SM120` (`99 KiB = 101,376 B`) 上启用 TMA？
   - *解答*：每个缓冲区元素数为 $2 \times 64 \times 128 = 16,384$。$S_{\text{grad}} = 16384 \times 2 = 32,768\text{ B}$，$2 S_{\text{in}} = 2 \times 16,384 = 32,768\text{ B}$，$2 S_{\text{out}} = 2 \times 16,384 = 32,768\text{ B}$。总共享内存 $= 32768 \times 3 + 128 + 96 = 98,528\text{ B} \le 101,376\text{ B}$，因此可在 `SM120` 上启用 TMA。
2. **可证伪假设**：在 Mooncake Store 的 `BatchProbeKey(..., "LastHitOnly", candidate_size=4)` 中，若最后一个候选组 $[k_8, k_9, k_{10}, k_{11}]$ 在阶段 1 探测全为 `1`，但在阶段 2 重验时 $k_8, k_9$ 已授予读租约而 $k_{10}$ 发现已被驱逐，Master 是否必须回滚 $k_8, k_9$ 的读租约才能保证正确性？
   - *证伪条件*：无需回滚。读租约仅提供有限 TTL 的驱逐保护（Soft-Pin/TTL），不改变键值内容；Master 将 `results[10]` 置为 `0`（使该候选组不再全 `1`）并继续向前重验锁定完整候选组 $[k_4, k_5, k_6, k_7]$，$k_8, k_9$ 上的短暂租约会在 TTL 到期后自然释放，无任何正确性风险。
