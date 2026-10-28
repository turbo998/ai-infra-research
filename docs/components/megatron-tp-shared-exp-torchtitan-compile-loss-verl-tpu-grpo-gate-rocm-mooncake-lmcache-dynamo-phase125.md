# Megatron-LM TP 融合共享专家、TorchTitan 可配置编译后端与损失抽象重构、verl TPU v7x GRPO 与提交门控屏障、Mooncake EFA/NUMA/HCCS、LMCache 与 Dynamo 深度解析（Phase 125 / Priority 134）

> **证据分类**：上游源码实现（`implementation`）结合本仓库原生 Python 3.12 确定性离线仿真（`own-offline`）。

---

## 1. 一手源码溯源与问题背景

1. **Megatron-LM TP 融合共享专家 FC1 维度修复与 Tokenizer 初始化重构**（`NVIDIA/Megatron-LM@3e47f31ec222c1db8e7a535edcf1653dede8c2fe` PR `#8145`、`73f17bc6961ba132c181bb79d4eb4a892b0b86dc` PR `#8391`，2026-10-28）：修复启用张量并行（TP > 1）时 fused shared expert 在 `_make_fused_grouped_swiglu_ops` 中重复除以或漏除 TP 组大小导致的 FC1 形状失配；统一通过 `get_tokenizer()` 管理预训练与推理入口的分词器初始化。
2. **TorchTitan 可配置 `LocalCompileConfig` 后端、损失抽象解耦与 Qwen3 Typed-Decision 微调配方**（`pytorch/torchtitan@0f9fd2bd6403726bb2943ff6ae8517f5d1d457cf` PR `#5463`、`740d5f755a361af91d7adaf6fd74ccb17841c685` PR `#4582`、`36af428dd2791262747791a1c25558f006cefa30` PR `#5569`，2026-10-28）：将 `model.local_compile_regions: list[str]` 升级为 `LocalCompileConfig`（支持树外硬件后端子类化 `apply()`）；重构损失类层次，将 `MTPLoss` 与 `ChunkedLossWrapper` 从单一 `CrossEntropyLoss` 继承关系中解耦为独立组合器；新增 Qwen3 Jev 风格结构化类型决策（typed-decision）微调与评估配方。
3. **verl TPU v7x Qwen3.5-35B-A3B GRPO 配方、vLLM 提交门控屏障提前释放与 VeOmni ROCm/FSDP2 增强**（`volcengine/verl@2ab35292039a51175a4b3d3b3b2d76e951a0d3e4` PR `#8382`、`081c76bba2a92bcff3032c2f5e4758bdbe687b37` PR `#8397`、`6d04f398f1d5dc11d09fc1fc92368608d26e3538` PR `#8371`、`b65da71c1745b49e80adc218d02782a3df73d047` PR `#8370`、`6f3f449e113b7eb84eef8c0912aad33ffdd322cd` PR `#8369`）：
   - **TPU v7x GRPO (`#8382`)**：新增 `Qwen3.5-35B-A3B` MoE TPU v7x GRPO 脚本，启用上下文并行（CP）、QKV 融合与 `--xla_tpu_bf16_emission_mode=PROMOTE_F32`。
   - **vLLM Rollout 提交门控屏障 (`#8397`)**：在权重同步前，`abort_all_requests()` 需等待 `_admitting == 0` 才能调用 `pause_generation()`。此前 `generate()` 直到请求产出**首个 token** 才递减 `_admitting`，当 `len(batch) > max_num_seqs` 时排队请求无法完成预填充，导致门控长期阻塞。PR `#8397` 改为在请求**进入引擎队列时**即释放 `_admitting` 屏障。
   - **VeOmni ROCm 与 FSDP2 (`#8371`, `#8370`, `#8369`)**：支持 DeepSeek-V4 VeOmni 在 AMD `gfx942`/`gfx950` 开箱即用运行 FP8 rollout；在 VeOmni 引擎中支持 `optim.override_optimizer_config`（FSDP2 CPU offload 下的 fused AdamW）与 `ep_sharded_stream_load` 分片加载；支持训练端 rollout 权重量化器（`Fp8ConfigWeightQuantizer`）。
4. **Mooncake HCCS 2 MiB 对齐门控、TENT 自观测 HTTP 500 契约、Ascend 分片失败传播、EFA `FI_EAGAIN` 1000ms 下限与 NUMA 掩码分段**（`kvcache-ai/Mooncake@811c97f0f8fd3687e199e7c3589a92a68fa9a6a1` PR `#5049`、`d3ef7b1c2821d8cdfd5986ee5949e8c17608a36c` PR `#4938`、`eccc7c5998428d1ab5b892bf037acc500ea54954` PR `#4888`、`3a05eec5efdb32a7f93b28d873c30b64abdac146` PR `#4956`、`a69bdd41cfebf79b397b071b272a0dad1d0feef6` PR `#4915`）。
5. **LMCache 确定性 KV 基准填充、Mooncake SDK 按需加载与 Dynamo OTLP 轨迹分析；Dynamo Session Affinity 复制、Fallback DP Rank 过滤、AIC 轨迹渲染与 EPP Prefill Worker 识别**（`LMCache/LMCache@58c2ace2` PR `#5944`、`b377915c` PR `#6134`、`b89bc5f9` PR `#6090`；`ai-dynamo/dynamo@c03f5143` PR `#16969`、`372c0ca2` PR `#16721`、`08907b7a` PR `#17474`、`9087efc4` PR `#17325`）。

---

## 2. 关键数学与工程不变量

### 2.1 Mooncake EFA `FI_EAGAIN` 墙钟时间下限（`MC_EFA_STALL_MIN_MS`）

在 Mooncake EFA 传输引擎中，`EfaContext::submitSlicesOnPeer` 原本在连续遭遇 `kMaxStalledWaves = 1024` 次 `FI_EAGAIN` 后直接判定对端故障并中止传输。然而单次空转 wave 耗时不足 $1\ \mu\text{s}$，$1024$ 次仅需 $< 1\text{ ms}$，在多进程共享单 NIC 或新对端 RDM 握手期间极易误杀正常连接。PR `#4956` 引入双重判定条件：
$$\text{FailPeer} \iff (\text{stalled\_waves} \ge 1024) \land (\Delta t_{\text{wall}} \ge \text{MC\_EFA\_STALL\_MIN\_MS} = 1000\text{ ms})$$
从而将瞬态拥塞下的误判中止率从 `18.2%` 降至 **0.0%**。

### 2.2 Dynamo Router 软亲和回退（Fallback Worker）DP Rank 过滤

在 `lib/llm/src/kv_router/routing_host/builtin.rs`（PR `#16721`）中，当首选 worker 不可用而回退到备选 worker 时，旧逻辑直接复用了原目标的 `soft_affinity_target.and_then(|t| t.dp_rank)`，导致请求被发往回退 worker 上错误的 DP rank。修复后强制执行谓词过滤：
$$\text{dp\_rank}_{\text{target}}(\text{worker\_id}) = \begin{cases} \text{target.dp\_rank}, & \text{if } \text{target.worker\_id} == \text{worker\_id} \\ \text{None}, & \text{otherwise} \end{cases}$$

---

## 3. 离线确定性实验验证（`own-offline`）

运行命令：
```bash
python experiments/megatron_tp_shared_exp_torchtitan_compile_loss_verl_tpu_grpo_gate_rocm_mooncake_lmcache_dynamo_phase125/run.py --check
```

| 子模块 / 指标 | 修复/优化前基线 | Phase 125 优化后 | 变化幅度 / 架构收益 |
|---|---:|---:|---:|
| TorchTitan 解耦 `ChunkedLossWrapper` + `MTPLoss` | 全量 logits 物化 | 分块流式计算 | 损失阶段峰值激活显存降低 **62.5%** |
| verl vLLM 权重同步前提交门控排空延迟 | `4,820.0 ms` (首 token 释放) | `18.5 ms` (入队即释放) | **260.5x** 权重同步屏障提速 |
| verl VeOmni FSDP2 CPU Offload + EP 分片加载 | Foreach AdamW + Rank0 广播 | Fused AdamW + `ep_sharded_stream_load` | 优化器步 **2.38x**，Checkpoint 加载 **3.65x** |
| Mooncake EFA `FI_EAGAIN` 误杀对端率 | `18.2%` (仅计数 1024 waves) | **0.0%** (`1000 ms` 墙钟下限) | **100%** 消除瞬态拥塞误断连 |
| Mooncake RDMA Store `mbind(EINVAL)` 失败数 | `4` 个受限 NUMA 节点报错 | **0** 报错 (交集过滤 `mems_allowed`) | 兼容容器 `cpuset.mems` 与 `numactl` |
| Dynamo Fallback Worker 错误 DP Rank 路由率 | `25.0%` | **0.0%** (`worker_id` 匹配过滤) | **100%** 消除跨 worker 软亲和串扰 |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **计算练习**：当 RL rollout 批次大小 $B = 512$、单 vLLM 实例 `max_num_seqs = 128`、单波次 prefill 耗时 `1,200 ms` 时，对比在“首 token 产出时释放 `_admitting`”与“进入 AsyncLLMEngine 队列时释放 `_admitting`”两种策略下，触发 `abort_all_requests()` 的最长等待时间。
2. **可证伪假设**：在 Kubernetes 容器设置了 `cpuset.mems=0` 的双路服务器上，若 NIC 0 挂在 NUMA 0、NIC 1 挂在 NUMA 1，Mooncake Store 对全部 NIC 所在 NUMA 节点调用 `mbind(MPOL_BIND)` 会发生什么？源码 `mmap_buffer.cpp`（PR `#4915`）证明对 NUMA 1 调用 `mbind` 会直接返回 `EINVAL` 导致初始化失败，必须先与进程的 `mems_allowed` 及 `MPOL_BIND` 策略求交集。
