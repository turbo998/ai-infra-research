# PyTorch ROCm `NCCLSymmetricMemory` (RCCL) 对称内存、TokenSwitch `topk_weights` 反向梯度、Inductor 编译期自动调优 TMA 描述符包装与 `KERNEL_CONFIGS` 持久化，Megatron-LM `get_tokenizer` 统一初始化，Mooncake EFA `88 GiB` 单 NIC 4KB 页上限与 2MB 巨页注册，LMCache 并发 TP Worker 基准传输，及 Dynamo Operator DCGM 连接关闭与 `SelectionService` 会话亲和性外部复制深度分析 (Phase 129 · Priority 138)

> **证据类型声明**：本报告结合 `pytorch/pytorch`（`b8c7d2c`, `c5c9a00`, `b8d6389`, `e5d366e`）、`NVIDIA/Megatron-LM`（`73f17bc`）、`kvcache-ai/Mooncake`（`e648c15`）、`LMCache/LMCache`（`d67052b`）与 `ai-dynamo/dynamo`（`f9c22d7`, `c03f514`）上游一手源码提交（`implementation`）及本项目 `experiments/pytorch_rccl_symm_tokenswitch_tma_mooncake_efa_lmcache_dynamo_phase129/run.py` 的 Python 3.12 标准库确定性离线算例（`own-offline`）。未宣称在真实 GPU/EFA 集群上完成真机重测。

---

## 1. 核心架构问题与背景

在跨平台分布式训练（PyTorch ROCm RCCL 对称内存、TokenSwitch EP 梯度、Inductor 模块级内核编译）与云原生分离式推理缓存（Mooncake EFA、LMCache、Dynamo）中，底层传输页表规模、编译期配置持久化与控制面资源生命周期是规模化稳定性的核心：

1. **PyTorch ROCm `NCCLSymmetricMemory` (RCCL) 与 TokenSwitch `topk_weights` 反向传播（`pytorch@b8c7d2c`, `c5c9a00`）**：适配 HIP 图捕获、RCCL 窗口注册与设备端规约 API，在 ROCm 上打通 `NCCLSymmetricMemory` 的分配、Rendezvous、图捕获与 `3` 类设备端单边/规约原语；在 `torch/csrc/distributed/c10d/symm_mem/nccl_ep.cu` 的 `TokenSwitch` EP 算子中补齐 `topk_weights` 反向梯度计算（$\nabla w_{t,k} = \langle \nabla y_t, e_{t,k} \rangle$，`4096×8×7168` 下 `234.88M` MACs），打通端到端路由器训练。
2. **PyTorch Inductor 编译期调优 TMA 描述符包装与 `KERNEL_CONFIGS` 持久化（`pytorch@b8d6389`, `e5d366e`）**：在 `triton.autotune_with_sample_inputs=True` 下将捕获的底层张量包装进宿主端 TMA 描述符调用，消除编译期调优块的签名类型失配崩溃；在模块级内核生成中写入 `KERNEL_CONFIGS` 字典持久化编译期选定的 Triton 配置，使运行时首次启动重调优从 **`64 -> 0` 个内核**，并消除跨 Rank 因二次调优选择不同规约顺序引发的数值漂移风险。
3. **Megatron-LM `get_tokenizer` 统一初始化（`megatron@73f17bc`）**：在 BERT 预训练与 GPT 动态推理控制器中统一通过 `global_vars.get_tokenizer` 获取分词器，消除同一 Rank 内的重复构建（`3 -> 1` 次/Rank）。
4. **Mooncake EFA `88 GiB` 4KB 页上限与 2MB 巨页注册机制（`mooncake@e648c15`）**：在 AWS EFA 网卡上，基于 4KB 普通页的单次内存注册（MR）每个 NIC 仅能覆盖约 **`88 GiB`**；当 SGLang/Mooncake 主机 KV 池达 **`128 GiB`** 时，单 NIC 主机直接注册失败，多 NIC 主机被迫跨 NIC 子集切分。采用 `hugetlb` 或 `MADV_HUGEPAGE`（THP 2MB 巨页，注意 `memfd` 需显式配置）将页表项压缩 **`512.0x`**（`33,554,432 -> 65,536` 项），使单 NIC 即可完整注册 `128 GiB+` 池。
5. **LMCache 并发 TP Worker 基准传输与 Dynamo DCGM 套接字回收 / 会话亲和性复制（`lmcache@d67052b`, `dynamo@f9c22d7`, `c03f514`）**：LMCache `bench` 并发执行所有已启用 TP Worker 的 `STORE`/`RETRIEVE` 并引入 `--transfer-timeout`（默认 `10.0 s`），在 `TP=8` 下将基准耗时从 `148.0 ms` 压降至 **`19.6 ms`（`7.55x` 提速）**；Dynamo Kubernetes Operator 在每次 DCGM 指标抓取后关闭连接并限制空闲连接池，消除 `64` 节点集群每小时 **`23,040 -> 0` 个泄漏套接字**；`SelectionService` 暴露 `SessionBindingStream` 供外部宿主实时复制会话绑定（故障切换亲和命中率 **`100%`**）。

---

## 2. 数学模型与确定性验证结果（`own-offline`）

| 评估维度 | 上游提交 | 旧版基线行为 | Phase 129 修复/优化后 | 关键量化收益 |
|---|---|---|---|---|
| **PyTorch ROCm 对称内存 & TokenSwitch** | `pytorch@b8c7d2c`, `c5c9a00` | ROCm 不支持 `NCCLSymmetricMemory`；`TokenSwitch` 无 `topk_weights` 梯度 | 打通 RCCL 对称内存 `3` 原语 + `nccl_ep.cu` `topk_weights` 反向梯度 | **支持 ROCm 单边通信与 `234.88M` MACs 门控梯度反传** |
| **PyTorch Inductor TMA & `KERNEL_CONFIGS`** | `pytorch@b8d6389`, `e5d366e` | 样本输入 TMA 传裸张量报错；首次启动丢弃编译期配置重调 `64` 个内核 | 包装 TMA 描述符 + 模块持久化 `KERNEL_CONFIGS` | **首次启动重调优 `64 -> 0` 个内核，跨 Rank 规约漂移风险 `0`** |
| **Megatron-LM 分词器初始化** | `megatron@73f17bc` | 多入口重复调用 `build_tokenizer`（`3` 次/Rank） | 统一使用 `get_tokenizer()` | **每 Rank 重复分词器初始化 `3 -> 1`** |
| **Mooncake EFA 巨页内存注册** | `mooncake@e648c15` | 4KB 页每 NIC 上限 **`88 GiB`**（`128 GiB` 单 NIC 失败） | `hugetlb` / THP `2 MiB` 巨页注册 | **MR 页表项减少 `512.0x`，单 NIC 完整注册 `128 GiB+`** |
| **LMCache 并发 TP Worker 传输** | `lmcache@d67052b` | `TP=8` 串行执行 Worker 传输（`148.0 ms`） | 并发执行全部 TP Worker + `--transfer-timeout`（`19.6 ms`） | **`TP=8` 传输吞吐基准提速 `7.551x`** |
| **Dynamo DCGM 抓取与会话亲和复制** | `dynamo@f9c22d7`, `c03f514` | DCGM 抓取后未关连接（`23,040` 泄漏/h）；无外部会话绑定流 | 每次请求后关闭连接 + 暴露 `SessionBindingStream` | **泄漏套接字 `23,040 -> 0/h`，外部副本故障切换命中 `100%`** |

---

## 3. 可复现验证命令

```bash
python experiments/pytorch_rccl_symm_tokenswitch_tma_mooncake_efa_lmcache_dynamo_phase129/run.py --check
python -m unittest tests/test_phase129_experiments.py -v
```
