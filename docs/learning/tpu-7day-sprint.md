# TPU 与 AI Hypercomputer — 7 天冲刺

从数据中心网络到 JAX：给云 / 网络架构师的一周加速器集群补课路线——系统全景、分布式训练、推理与性能剖析、集群控制面、机房落地与上电开通。

**时间预算**: 约 20 小时：工作日晚上每天 2–2.5 小时，一个低强度阅读夜（1.5 小时），一个 6 小时动手大块。

> 网页版：https://turbo998.github.io/workshops/tpu-7day-sprint/index.html

## 一周安排

| 第 | 主题 | 时长 |
|---|---|---|
| 1 | TPU 系统全景，以及 TPU 正在部署到哪里 | 2.5h |
| 2 | TPU 上的分布式训练：JAX、XLA 与分片 | 2.5h |
| 3 | TPU 推理与性能剖析 | 2.5h |
| 4 | 集群控制面：TPU VM、GKE、Slurm 与存储 | 2.5h |
| 5 | 物理层：机房资格认证、上电开通与运维就绪 | 1.5h (reading) |
| 6 | 动手实验日 | 6h (hands-on) |
| 7 | 整合：从头到尾讲清楚 | 2.5h |

### Day 1 — TPU 系统全景，以及 TPU 正在部署到哪里

*建立 芯片 → 主机 → cube → slice → pod → multislice 的心智模型，并理解 TPU 走出 Google 自有机房、进入伙伴与客户数据中心的趋势。*

**学什么**
- 芯片：MXU 脉动阵列、TensorCore 与 SparseCore 的分工、HBM；TPU7x（Ironwood）每芯片 2 个 TensorCore + 4 个 SparseCore，采用双 chiplet 编程模型
- 代际与定位：v5e（性价比）、v5p（训练，8,960 芯片 pod）、v6e Trillium（256 芯片 pod）、TPU7x Ironwood（9,216 芯片 superpod，单芯片 BF16 2,307 TF / FP8 4,614 TF）、第 8 代拆分为 TPU 8t（训练）与 TPU 8i（推理）
- 拓扑术语：4x4x4 cube、只有满 cube 才有回环链路的 3D torus、accelerator type 与 accelerator config 两种写法、单主机与多主机 slice（TPU7x 每台 VM 4 颗芯片）
- 互连：slice 内走 ICI，cube 之间走光路交换（OCS），ICI resiliency 可绕开光链路故障，Multislice 跨 slice 走数据中心网络（DCN）
- 市场背景：TPU 系统开始部署到非 Google 运营的机房——由伙伴负责现场安装、布线、老化测试与验收，出现 TPU-as-a-service 合资公司，以及面向机密 / 主权场景的客户本地部署


**读什么**
- [TPU system architecture (docs)](https://docs.cloud.google.com/tpu/docs/system-architecture)
- [TPU7x (Ironwood) (docs)](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Inside the Ironwood TPU co-designed AI stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)
- [Eighth-generation TPUs: TPU 8t & 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)
- [How to Scale Your Model — ch. 1–2 (rooflines, TPUs)](https://jax-ml.github.io/scaling-book/)

**当天产出**: 一页速查表：凭记忆画出层级图 + 一张 v5e / v5p / v6e / TPU7x / 8t / 8i 对比表（pod 规模、单芯片算力、HBM、目标负载）。

**自测题**
- [ ] 一个 2-slice 作业里，哪些流量走 ICI，哪些走 DCN？
- [ ] 4x4x8 和 4x4x4 的 slice 行为有什么不同？
- [ ] 某个 OCS 端口断了，什么会失败，什么还能继续跑？


### Day 2 — TPU 上的分布式训练：JAX、XLA 与分片

*把各种并行策略映射到 device mesh 上，知道哪个维度该放在 ICI、哪个放在 DCN。*

**学什么**
- 并行方式：数据并行、FSDP（ZeRO-3 风格）、张量并行、流水线并行、专家并行、序列 / 上下文并行，以及各自对应的集合通信（all-reduce、all-gather、reduce-scatter、all-to-all）
- JAX 基础：jax.jit、Mesh / NamedSharding / PartitionSpec、shard_map，XLA 的 GSPMD → Shardy 分区器
- 经验法则：带宽敏感的维度（张量并行、FSDP）放在 ICI；跨 slice 的数据并行走 DCN
- 参考栈：MaxText（JAX 开源 LLM 训练）、PyTorch 路线（PyTorch/XLA、torchax）、超大多 slice 作业用 Pathways
- 估算：训练 FLOPs ≈ 6 × 参数量 × token 数；MFU = 实际 FLOPs / 峰值 FLOPs


**读什么**
- [How to Scale Your Model — sharding & transformer math chapters](https://jax-ml.github.io/scaling-book/)
- [JAX: distributed arrays & automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [Cloud TPU Multislice overview](https://docs.cloud.google.com/tpu/docs/multislice-introduction)

**当天产出**: 一道完整估算：在 TPU7x 上以 40% MFU、30 天训练 70B dense 模型（15T token）需要多少芯片，以及你会提议的 mesh 布局。

**自测题**
- [ ] 70B 模型为什么通常用 FSDP 而不是纯数据并行？
- [ ] 张量并行主要是哪种集合通信？为什么必须留在 ICI 上？
- [ ] 把 PyTorch 训练循环迁到 TPU 时，需要改什么？


### Day 3 — TPU 推理与性能剖析

*在 TPU 上服务 LLM，按权重 + KV cache 估算 HBM，用 profiler 定位瓶颈。*

**学什么**
- vLLM TPU（tpu-inference）：JAX 模型和 PyTorch（torchax）模型统一走 JAX 下沉路径；推荐代际 v5e / v6e / TPU7x；支持 chunked prefill 和基于 DCN 的 prefill/decode 分离
- Prefill 受算力约束，decode 受内存带宽约束；batch size、量化（int8 / FP8）和 KV cache 决定每 token 成本
- KV 估算：每 token 字节数 = 2 × 层数 × KV 头数 × head_dim × 字节数；例如 70B GQA 模型（80 层、8 个 KV 头、d=128、BF16）约 0.33 MB/token
- 性能剖析：jax.profiler → XProf（trace viewer、HLO op profile、step 时间拆解）；判断一个 step 是算力、内存、集合通信还是输入（infeed）瓶颈
- 自定义 kernel：TPU 上的 Pallas（如 ragged paged attention）——什么时候值得写 kernel，什么时候改一个 flag 或分片就够


**读什么**
- [vLLM TPU: unified backend for PyTorch and JAX](https://vllm.ai/blog/vllm-tpu)
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)
- [XProf](https://github.com/openxla/xprof)
- [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)

**当天产出**: 一棵瓶颈排查树（现象 → profiler 证据 → 修复手段），以及服务 70B 模型、32K 上下文时的 HBM 预算。

**自测题**
- [ ] 70B BF16 模型权重 140 GB，再加 32 路并发 32K 序列，需要几颗芯片？
- [ ] profile 显示 step 之间有长空隙，你首先怀疑什么？
- [ ] 什么情况下会选 P/D 分离？


### Day 4 — 集群控制面：TPU VM、GKE、Slurm 与存储

*搞清楚作业怎样调度到加速器上、算力怎样拿到、训练怎样扛过故障。*

**学什么**
- 资源获取：TPU VM 与 queued resources；容量模式（预留、flex-start、spot）与长期预留
- GKE：TPU 节点池（单主机 / 多主机）、JobSet 负责多主机作业的 gang 启动、Kueue（含拓扑感知调度）负责配额与队列、GKE 上的 Multislice
- Slurm：用 Cluster Toolkit 在 Google Cloud 上搭 Slurm，面向 HPC 背景的研究团队
- Linux 主机层：libtpu 运行时、用 tpu-info 看芯片利用率 / HBM、驱动与镜像管理
- 数据与检查点：Cloud Storage + GCS FUSE、Hyperdisk ML、Managed Lustre；Orbax 异步检查点；goodput = 有效训练时间 / 总时间
- 实战案例（GPU，同一套控制面）：DWS flex-start 生命周期——约 10 分钟预订窗口、consume-provisioning-request 注解、7 天上限；label 不是调度约束（用 topology.kubernetes.io/zone 钉住可用区）；网络 fabric 不能跨可用区时用预留 + COMPACT placement


**读什么**
- [TPUs on GKE (docs)](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus)
- [JobSet](https://github.com/kubernetes-sigs/jobset)
- [Kueue](https://kueue.sigs.k8s.io/)
- [Cluster Toolkit (Slurm)](https://github.com/GoogleCloudPlatform/cluster-toolkit)
- [TPU machine types & capacity](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)
- [Case study: hypercomputer-training-jobs — DWS, Kueue, DCGM, NCCL smoke on GKE](https://github.com/turbo998/hypercomputer-training-jobs)

**当天产出**: 一张 GKE 上 2-slice 训练作业的参考架构（Kueue + JobSet + Multislice + GCS FUSE + Orbax），以及一张 GKE vs Slurm 选型表。

**自测题**
- [ ] 多主机 TPU 作业为什么需要 gang 调度？
- [ ] 训练到第 30 小时，slice 里一台主机挂了——讲一遍恢复过程。
- [ ] 什么情况下 Slurm 比 GKE 更合适？


### Day 5 — 物理层：机房资格认证、上电开通与运维就绪

*搞清楚把一个加速器 pod 落到别人运营的数据中心里，需要具备什么。*

**学什么**
- 机房资格认证：供电容量与冗余、机柜功率密度、液冷（CDU、设施水回路、漏液检测）、地板承重、光纤与 DCN 上联、物理安全、物流
- Pod 布局：cube 与机柜摆放、到 OCS 的光纤布线、把供电与制冷分区当作故障域
- 上电开通：上架、布线校验、老化测试、固件基线、ICI 链路健康、集合通信带宽测试，用一个参考训练作业作为验收门槛
- 运维就绪：带外管理（BMC、Redfish / IPMI）、遥测进 Prometheus（主机、加速器、机房设施）、告警、硬件维修与 RMA 流程、备件、远程协助（remote hands）SLA
- 为什么关系到 goodput：故障域 + 备件 + 快速维修 + 检查点 = 真正可用的算力


**读什么**
- [Ironwood GA announcement (liquid cooling, fleet uptime)](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads/)
- [DMTF Redfish](https://www.dmtf.org/standards/redfish)
- [Open Compute Project](https://www.opencompute.org/)
- [SemiAnalysis: TPUv7 and TPU externalization](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the)

**当天产出**: 只做阅读笔记——明天把它们整理成上电开通与运维就绪清单。

**自测题**
- [ ] 液冷 pod 发货到机房前，必须确认哪五件事？
- [ ] 从“上架完成”到“交付客户”之间的验收门槛是什么？


### Day 6 — 动手实验日

*真正上手 TPU：测量、剖析，并写出运维产出物。*

**学什么**
- 实验 A（1.5h）——免费 TPU 运行时（Kaggle 或 Colab，先看当前提供的 TPU 型号）：列出设备、建 mesh、跑一个分片的 BF16 矩阵乘，算出实际 TFLOP/s 和峰值的差距
- 实验 B（1.5h）——用 jax.profiler 抓 trace，在 XProf 里打开，给 step 的每一段贴标签（计算 / 集合通信 / 空闲）
- 实验 C（1.5h，有云额度再做）——在单主机 v6e 上用 vLLM TPU 服务一个小模型（如 Qwen3-4B），记录 3 种 batch size 下的 TTFT 和 tokens/s
- 实验 D（1.5h）——写出 TPU 机房上电开通与运维就绪清单，以及 Prometheus 指标目录（主机 / 加速器 / 带外 / 机房设施）
- 实验 E（可选，有云额度再做）——在小规模 GKE 节点池（L4 GPU 或单主机 TPU）上复现 DWS flex-start 流程：ProvisioningRequest + 消费它的 pod，再配一套 Kueue ResourceFlavor / ClusterQueue / LocalQueue + AdmissionCheck；按复盘格式写一页笔记


**实验 A/B 起步代码**

```python
import time, jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

devs = jax.devices(); n = len(devs); print(devs)
mesh = jax.make_mesh((n,), ("x",))
N = 8192
a = jax.device_put(jnp.ones((N, N), jnp.bfloat16), NamedSharding(mesh, P("x", None)))
b = jax.device_put(jnp.ones((N, N), jnp.bfloat16), NamedSharding(mesh, P(None, None)))
f = jax.jit(lambda a, b: a @ b)
f(a, b).block_until_ready()                      # compile + warm-up
t = time.perf_counter()
for _ in range(10): c = f(a, b)
c.block_until_ready()
dt = (time.perf_counter() - t) / 10
tf = 2 * N**3 / dt / 1e12
print(f"{tf:.1f} TFLOP/s total, {tf/n:.1f} per device")

jax.profiler.start_trace("/tmp/trace")           # Lab B: open with XProf
for _ in range(5): f(a, b).block_until_ready()
jax.profiler.stop_trace()
```


**读什么**
- [vLLM TPU quickstart (tpu-inference)](https://github.com/vllm-project/tpu-inference)
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)

**当天产出**: 一份带数字的实验记录（TFLOP/s、标注过的 trace 截图、可选的 TTFT 表），外加两份运维产出物：上电开通清单和指标目录。

**自测题**
- [ ] 你测得的 TFLOP/s 为什么低于峰值？trace 里哪一段能证明？


### Day 7 — 整合：从头到尾讲清楚

*把这一周的内容变成能在白板前 10 分钟讲完的一个完整故事。*

**学什么**
- 白板场景：某地区客户想在伙伴数据中心部署 4,096 颗 TPU7x——资格认证 → pod 布局 → 上电开通 → 负载迁移 → 运维
- GPU → TPU 迁移叙事：哪些容易迁（JAX、vLLM 服务的模型），哪些要下功夫（自定义 CUDA kernel），怎样用 PoC 降低风险
- 补上第 1–6 天发现的漏洞；不看笔记重做所有自测题


**读什么**
- [How to Scale Your Model — inference & practice chapters](https://jax-ml.github.io/scaling-book/)

**当天产出**: 一段 10 分钟的白板讲解，自己录一遍再回看。

**自测题**
- [ ] 能不能用大白话给 CIO 讲清楚 ICI vs DCN、MFU、KV 估算和上电验收门槛？


## 算例

- **训练估算（第 2 天）**: 6 × 70e9 × 15e12 ≈ 6.3e24 FLOPs。TPU7x 峰值 2,307 TF BF16 × 40% MFU ≈ 0.92 PF/芯片 ≈ 8.0e19 FLOPs/芯片·天 → 约 79,000 芯片·天 → 30 天约需 2,600 颗芯片。
- **KV 估算（第 3 天）**: 2 × 80 × 8 × 128 × 2 B ≈ 0.33 MB/token → 32K 上下文每条序列约 10.7 GB → 32 路并发约 344 GB，再加 140 GB BF16 权重，未计激活与开销已约 484 GB。
- **解读**: 光权重就超过单芯片 HBM，所以服务时需要跨多颗芯片做张量并行（或用 FP8 权重）——并发上限由 KV cache 决定，而不是算力。


## 上电开通与运维就绪清单

**机房资格认证**
- [ ] 按目标机柜功率密度确认供电容量与冗余
- [ ] 液冷：CDU 容量与冗余、设施水规格、漏液检测
- [ ] 地板承重、机柜占地与维护通道
- [ ] 光纤、DCN 上联与交叉连接
- [ ] 物理安全与门禁；进口 / 物流路径

**上电开通**
- [ ] 按批准的 pod 布局上架
- [ ] 端到端校验光纤布线（ICI / OCS / DCN）
- [ ] 推送固件与软件基线
- [ ] 通过老化测试与硬件诊断
- [ ] ICI 链路健康与集合通信带宽达标（GPU 上的对应做法：DCGM 诊断 + NCCL all_reduce_perf 冒烟门槛）
- [ ] 参考训练作业达到目标 step 时间——作为验收门槛

**运维就绪**
- [ ] 带外管理可达（BMC，经 Redfish / IPMI）
- [ ] 遥测进入 Prometheus：主机、加速器、带外传感器、机房设施（CDU / PDU）
- [ ] 告警路由与升级矩阵已确认
- [ ] 硬件维修 / RMA 流程、备件池、远程协助 SLA
- [ ] 演练过检查点与重启 runbook；定义 goodput 目标


## 背景：TPU 正在走向哪里

- [Ironwood GA (Nov 2025)](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads/)
- [TPU 8t / 8i (Apr 2026)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)
- [TPUs offered to AI cloud providers (DCD)](https://www.datacenterdynamics.com/en/news/google-offers-its-tpus-to-ai-cloud-providers-report/)
- [TPUs in customers’ own data centers (Nov 2026)](https://fool.com/investing/2026/11/12/should-you-buy-alphabet-stock-now-that-google-will-let-customers-run-its-ai-chips-in-their-own-data-centers)
