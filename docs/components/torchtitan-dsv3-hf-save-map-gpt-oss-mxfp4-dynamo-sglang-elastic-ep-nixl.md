# TorchTitan DeepSeek-V3 HF 保存索引过滤与 GPT-OSS 双布局适配器、Dynamo SGLang Elastic-EP NIXL 首轮 MoE 前向与 TRT-LLM OpenMPI `ob1` 快照传输，及 Mooncake 元数据访问器 / NIXL Trace V2 深度剖析

> **Lang / 语言**: **中文 (当前)** | [English](./torchtitan-dsv3-hf-save-map-gpt-oss-mxfp4-dynamo-sglang-elastic-ep-nixl.en.md)
>
> **证据等级 (Evidence Class)**：`implementation`（上游固定 Commit 源码核验）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/torchtitan_dsv3_hf_save_map_gpt_oss_mxfp4_dynamo_elastic_ep/run.py --check`）。

---

## 1. 架构背景与核心问题

在超大规模 MoE 训练与推理的检查点互操作（TorchTitan <-> HuggingFace Safetensors）、在线弹性专家并行扩缩容（Dynamo + SGLang Elastic-EP over NIXL）、进程快照恢复（TRT-LLM CRIU Snapshot/Restore）以及分布式 KV 存储控制面（Mooncake Master + NIXL Tracing）中，2026-10-20 上游合入的 6 项核心提交解决了以下生产级故障模式：

1. **TorchTitan DeepSeek-V3 671B HF 检查点导出损坏（`unknown variant ''`）**：`DeepSeekV3StateDictAdapter` 从开源发布的 `deepseek-ai/DeepSeek-V3` 的 `model.safetensors.index.json`（共 `91,991` 个条目）读取 `fqn_to_index_mapping`。由于 DCP `HuggingFaceStorageWriter` 合并阶段会为映射表中的**每一个键**生成输出文件头，任何未被 TorchTitan 实际写出的键都会在 safetensors 头部生成 `{"shape": [], "dtype": ""}`，导致后续读取直接报错 `unknown variant ''` 拒绝打开文件。开源索引中有三类差异：
   - `45,808` 个 FP8 `X.weight_scale_inv` 缩放张量（TorchTitan 保存未量化权重，从不写出）；
   - 当 `num_mtp_layers=0` 时，第 61 层 MTP 层 `model.layers.61.*` 的 `788` 个张量；
   - 当 `num_mtp_layers=1` 时，`model.layers.61.embed_tokens.weight` 与 `model.layers.61.shared_head.head.weight` 与主模型共享内存，若直接导出同一 Tensor 引用会被 `safetensors` 因同文件别名（Tensor Alias）拒绝序列化。
2. **TorchTitan GPT-OSS 专家权重 BF16 与 MXFP4 双布局转置失配**：TorchTitan 内部按 `nn.Linear` 习惯（输出维度在前）存储路由专家权重 `w13.weight [E, 2, F, D]` 与 `w2.weight [E, D, F]`。开源 MXFP4 检查点（`gate_up_proj_blocks` / `down_proj_blocks`）反量化后同样是输出维度在前，但未量化 BF16 HF 检查点（如 `unsloth/gpt-oss-20b-BF16`）因 `transformers` 计算 `x @ W` 而采用**输入维度在前**的 `gate_up_proj [E, D, 2F]`（`[32, 2880, 5760]`）与 `down_proj [E, F, D]`。旧版 `to_hf` 无条件输出 `_blocks` 键名与 MXFP4 布局，导致 BF16 检查点既无法加载也无法正确导出。
3. **Dynamo SGLang Elastic-EP over NIXL 首轮 MoE 前向崩溃与扩容挂起**：在 SGLang 0.5.21/0.5.22 下，未量化 MoE 专家通过 NIXL 分发时默认落入已废弃的 `forward_deepgemm_masked` 路径导致 Worker 崩溃重启；同时 NIXL 分发器默认单 Rank 预算仅 128 tokens（遇到 1024 预填充块即刻 `abort`），且若未显式设置 `--moe-dense-tp-size 1` 或新加入节点（Joiner）与主节点参数不一致，扩容准入后的第一次前向即挂死。
4. **Dynamo TRT-LLM 快照恢复（Snapshot Mode）后首次集合通信断连**：OpenMPI 默认的 `ucx` PML 传输在 Rank 之间启用了 TCP Keepalive 探测。当快照生命周期冻结各 Rank 并停止网络流量时，未响应的 Keepalive 探测导致 Linux 内核直接断开连接，恢复后的引擎在第一次 MPI 集合通信即宣告失败。
5. **Mooncake Master 并发任务 ABA 元数据竞争与 NIXL 跨线程全量 TraceContext 传递**：Mooncake Store 需要按每键访问器契约（Per-Key Accessor Contract）与分片单调代数（`route::Generation`）隔离同一 Key 的不同发布生命周期；NIXL 则需要将 `TraceBackend` 从仅传递 64 位 `correlationId` 升级为传递完整 `TraceContext`（API V2），供 Chakra/OTel 等导出型后端读取 `trace_id`、`span_id` 与 `sampled` 标志。

---

## 2. 一手源码实现路径剖析

### 2.1 TorchTitan DeepSeek-V3 HF 索引过滤与 MTP 共享副本克隆（Commit `1d581795cd`）

在 `pytorch/torchtitan` Commit [`1d581795cd`](https://github.com/pytorch/torchtitan/commit/1d581795cd014a46e871cc9fc605325fbbceee85)（PR `#5065`，2026-10-20）中，`torchtitan/models/deepseek_v3/state_dict_adapter.py` 实现了精确的保存索引过滤与 MTP 副本克隆：

$$
N_{\text{saved}}(M) = 91{,}991 - 45{,}808 - (1 - M) \times 788 = \begin{cases} 45{,}395, & M = 0 \\ 46{,}183, & M = 1 \end{cases}
$$

- **索引过滤**：`__init__` 遍历 `self.fqn_to_index_mapping`，剔除所有 `.endswith(".weight_scale_inv")` 以及层号 $\ge \text{len(layers)} + \text{len(mtp\_layers)}$ 的键。
- **MTP 独立存储克隆**：`to_hf()` 通过 `state_dict[tt_key].clone()` 为每个 MTP 层生成独立的 `model.layers.{61+i}.embed_tokens.weight` 与 `shared_head.head.weight` 副本（671B BF16 下两个张量共 $2 \times 129280 \times 7168 \times 2 \approx 3.71\text{ GB}$），而 `from_hf()` 则跳过这些序列化副本，直接复用主模型的 `tok_embeddings.weight` 与 `lm_head.weight`。`DeepSeekV4StateDictAdapter` 因使用自有 `mtp.N.*` 命名，设置 `_MTP_SHARED_COPIES = {}` 退出该逻辑。

### 2.2 TorchTitan GPT-OSS BF16 / MXFP4 双布局适配器（Commit `da54a64ede`）

在 `pytorch/torchtitan` Commit [`da54a64ede`](https://github.com/pytorch/torchtitan/commit/da54a64ede6983f52f5d66a1290eefc5e13e3260)（PR `#5064`，2026-10-20）中，`torchtitan/models/gpt_oss/state_dict_adapter.py` 维护 `from_hf_map`（默认 BF16：`gate_up_proj` / `down_proj`）与 `from_hf_map_quantized`（MXFP4：`gate_up_proj_blocks` / `down_proj_blocks`）：
- **`to_hf()`**：先执行 `value.transpose(1, 2).flatten(1, 2)` 将 `w13 [E, 2, F, D]` 的 gate 与 up 行交错合并为 `[E, 2F, D]`；若 `not to_quantized_layout`（默认导出 BF16），再执行 `value.transpose(1, 2)` 转换为输入维度在前的 `[E, D, 2F]` 与 `down_proj [E, F, D]`。
- **`from_hf()`**：若 `not is_quantized_layout`，先执行 `value.transpose(1, 2)` 恢复输出维度在前，再执行 `value.unflatten(1, (-1, 2)).transpose(1, 2).contiguous()` 还原 `[E, 2, F, D]`。
- **索引重映射**：将 `model.safetensors.index.json` 中的 `X_blocks` 键剥离后缀映射为 `X`，并剔除 `X_scales`。

### 2.3 Dynamo SGLang Elastic-EP NIXL 不变量与 TRT-LLM `ob1` 快照传输（Commits `02da7f4486`, `c8b3a8e0e5`）

1. **SGLang Elastic-EP over NIXL（`ai-dynamo/dynamo` Commit [`02da7f4486`](https://github.com/ai-dynamo/dynamo/commit/02da7f44861b65758b960770e0763e9faa8e4ca4)，PR `#16865`）**：在主节点（Primary）与动态扩容加入组（Joiner，`run_sglang_elastic_ep_scale_up_test.sh` & `sglang_elastic_ep.yaml`）上强制对齐 5 项配置：
   - `SGLANG_DEEPEP_BF16_DISPATCH=1`、`SGLANG_NIXL_EP_BF16_DISPATCH=1` 与 `--moe-runner-backend deep_gemm`（规避 SGLang 0.5.21/0.5.22 已移除的 `forward_deepgemm_masked` 崩溃）；
   - `SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` 配合 `--chunked-prefill-size 1024`（防止超过默认 128 tokens 预算触发 NIXL dispatcher abort）；
   - `--moe-dense-tp-size 1`、关闭 Prefill/Decode CUDA Graph，并放宽 K8s Liveness Probe 容忍扩容期间数秒的内核重编译停顿。
2. **TRT-LLM 快照模式 OpenMPI `ob1` 传输（`ai-dynamo/dynamo` Commit [`c8b3a8e0e5`](https://github.com/ai-dynamo/dynamo/commit/c8b3a8e0e5cfb07199016ffde6499a384817e147)，PR `#16372`）**：在 `components/src/dynamo/trtllm/__main__.py` 导入 `dynamo.trtllm.main`（触发 OpenMPI 初始化）**之前**检测 `is_snapshot_enabled()`，预设 `OMPI_MCA_pml=ob1` 与 `OMPI_MCA_btl=tcp,self`，关闭 UCX 的 TCP Keepalive 探测。

### 2.4 Mooncake 每键元数据访问器与 NIXL Trace Plugin API V2（Commits `cc6b43e989`, `56c1c2f706`）

- **Mooncake Store（Commit [`cc6b43e989`](https://github.com/kvcache-ai/Mooncake/commit/cc6b43e9898c690e55a5e0c0da525e96ccb4d0ca)，PR `#4717`）**：在 `MasterService` 的 1024 个元数据分片（`kNumShards = 1024`）中引入受分片锁保护的单调递增代数 `route::Generation next_generation = 1`，每个 `ObjectRecord` 绑定唯一 `(generation, metadata, state)`，使对象元数据与其在途异步任务同生共死，消除同名 Key 驱逐后重新写入的 ABA 悬挂修改。
- **NIXL Trace Plugin API V2（Commit [`56c1c2f706`](https://github.com/ai-dynamo/nixl/commit/56c1c2f7062c31a69a1c08a1d2b3cc444c8e834b)，PR `#2445`）**：`TraceBackend` 虚接口升级为 `pushContext(const TraceContext &)` / `popContext()`，`nixl_trace_plugin_api_version` 升至 `V2`（`traceLoader` 严格拒绝 `V1` 旧插件）。`postXferReq`、`xfer.complete` 与 `TracerPhaseSink::recordPhase` 均传递完整 `TraceContext`，NVTX 后端从中提取 `context.correlationId64()` 保持时间线兼容，而导出型后端可直接格式化标准 W3C `traceparent`。

---

## 3. 确定性实验结果验证（`own-offline`）

运行 `python experiments/torchtitan_dsv3_hf_save_map_gpt_oss_mxfp4_dynamo_elastic_ep/run.py --check` 验证了全部 5 项核心不变量：
- DeepSeek-V3 索引过滤后条目数精确匹配写出张量数（`45,395` @ `M=0`、`46,183` @ `M=1`，空 safetensors 头数量从 `46,596` / `45,808` 降至 `0`）；
- GPT-OSS `w13` 在 TorchTitan `[E, 2, F, D]`、MXFP4 `[E, 2F, D]` 与 HF BF16 `[E, D, 2F]`（`[32, 2880, 5760]`）之间的双向往返无损还原；
- SGLang Elastic-EP over NIXL 首轮 MoE 前向与 TRT-LLM `ob1` 快照恢复后首次集合通信验证通过；
- Mooncake 跨代句柄拒绝（`stale_gen_1_rejected = True`）与 NIXL V1 插件拦截验证通过。

---

## 4. 面向解决方案架构师的设计准则与可证伪问题

1. **分布式检查点导出必须保证 `index.json` 映射集等于实际写出张量集**：复用开源量化版 `model.safetensors.index.json` 导出 BF16 权重时，必须先剔除 `.weight_scale_inv` / `_scales` 等量化附属键及未启用层（如 MTP），否则必然产生空 Header 损坏文件。
2. **容器/CRIU 快照恢复环境必须禁用带内核保活探测的 RDMA/UCX 控制通道**：在冻结-恢复窗口期内，任何开启底层 TCP Keepalive 的 MPI PML（如 UCX）都会在恢复后首轮集合通信时触发连接重置。
3. **后续可证伪问题**：在 SGLang Elastic-EP 扩容过程中，将 `--chunked-prefill-size` 从 `1024` 提升至 `4096` 时，增大 `SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 对 Mooncake RDMA 缓冲区注册内存占用的斜率是多少？
