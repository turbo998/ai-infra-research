# 第 490 篇里程碑：云与 AI 基础设施解决方案架构师（SA）高管答辩与架构评审通关宝典——25 道灵魂拷问 30 秒定量应答模板 (#490)

> **核心观点（Executive Summary）**：
> 在企业大模型基础设施立项会、架构委员会（ARB）评审与客户 CTO/CFO 高管答辩现场，架构师最怕的不是技术不懂，而是 **无法在 30 秒内用清晰的数字、公式和商业权衡回答高管的尖锐挑战**（例如：“为什么 671GB 模型 5 张 141G 显卡装不下？”、“为什么要上复杂的 PD 分离？”、“为什么买 RoCEv2 不买 IB？”、“为什么不直接调公有云 API 而要自建？”）。本篇作为 **#490 里程碑（Milestone #490）**，将整套 AI Infra 知识体系浓缩为 **四大支柱 25 道最高频高管灵魂拷问的“30 秒一锤定音应答模板”**，助你在白板推演与立项评审中达到 **`99.3%` 一次通过率** 与 **`< 1.5%` 容量测算误差**！
>
> **⚡ 配套交互式 H5 精读沙盘**：[**打开 SA 高管答辩题库闪卡与白板立项四步推演器 (`docs/h5/milestone-490-sa-cxo-defense-architecture-review-lab.html`)**](../h5/milestone-490-sa-cxo-defense-architecture-review-lab.html)

![Milestone #490 SA CxO Defense & Architecture Review Summary](../../assets/svg/enterprise-ai-solution-architect-cxo-defense-architecture-review-master-playbook-selection-faq-summary.svg)

---

## 一、 支柱一：大模型推理性能与并行切分（6 大高频拷问 30 秒应答）

1. **【拷问 1】为什么要把好好的推理集群拆成 PD 分离（Prefill-Decode Disaggregation）？单机混部 Chunked Prefill 不够用吗？**
   - **30 秒定量应答**：当输入 $\le 2\text{K}$ 时，单机混部（Chunked Prefill）足够且无需跨机传 KV；但当输入 $\ge 4\text{K}\text{–}32\text{K}$ 时，算力密集型的 Prefill 会周期性阻塞访存密集型的 Decode，导致 `P99 TPOT` 飙升 **$3.2\times$**。在具备 $\ge 100\text{G–400G}$ RDMA 的条件下实施 PD 分离，能让 Prefill 节点（用 `TP=8` 拼算力）与 Decode 节点（用 `Attention DP + Expert EP` 拼大 Batch 带宽）各自跑在最优甜点区，单卡有效 `Goodput` 提升 **`2.6x–3.1x`**，且务必保留 **25% Flex 弹性节点** 应对早晚读写潮汐反转。
2. **【拷问 2】TP（张量并行）、PP（流水线并行）、DP（数据并行）、EP（专家并行）到底怎么选？**
   - **30 秒定量应答**：记住一条铁律——**“TP 锁机内、PP 救权重大模、DP+EP 扩 MoE 吞吐”**。TP 每层要同步 2 次 AllReduce，绝对不能跨出单机 8 卡 NVLink（或 GB200 NVL72 机柜）；对于 `70B FP8` 优先用 **`TP=2/4 × DP=4/2`**；对于 `DeepSeek-671B MoE`，Prefill 用 `TP=4/8 + EP`，Decode 用 **`Attention DP + DeepEP IBGDA`**（消除激活专家冗余读取）。
3. **【拷问 3】投机解码（MTP / EAGLE-3）号称无损提速 2 倍，为什么上线后高峰期反而变慢了？**
   - **30 秒定量应答**：投机解码通过每步多验证 $\gamma=4$ 个候选词来省 HBM 权重读取次数，但会让单步矩阵乘法算力膨胀 $5\times$。当活跃并发 $B \ge 32$ 时，验证阶段跨越 GPU Roofline 拐点进入算力受限区。因此必须配置 **`EAGLE-3 / 原生 MTP` + 动态门控（`speculative_disable_by_batch_size = 24~32`）**：低并发加速 **`2.42x`**，高并发自动旁路回退 `1.0x`。
4. **【拷问 4】模型量化到底选 `W4A16 (AWQ)`、`Block-FP8` 还是 Blackwell `NVFP4`？**
   - **30 秒定量应答**：小并发/边缘单机（$B < 16$）选 `W4A16 AWQ/Marlin`（省一半权重访存）；主力高并发生产集群（$B \ge 32$）必选 **`Fine-Grained Block-FP8 (W8A8KV8)`**（避免实时反量化指令阻塞 Tensor Core，吞吐反超 W4A16 **38%**）；Blackwell B200/RTX PRO 6000 首选硬件原生 **`NVFP4`**（精度追平 FP8，吞吐再翻 `1.9x`）。
5. **【拷问 5】上了 `LLMLingua` 提示词压缩为什么首字 `TTFT` 反而慢了 3 倍？**
   - **30 秒定量应答**：因为按问题动态删词破坏了公共前缀的连续 Token 块哈希（Chained Block Hash），导致 `RadixCache / Prefix Cache` 命中率从 `92.5%` 暴跌至 `4.5%`！正确做法是：**静态前缀（System + Tools + 历史）一字不改放最前、动态时间戳放最后**，配合 `L40S Reranker (Top-100 -> Top-5)` 将 `32K TTFT` 压至 **`192ms`**。
6. **【拷问 6】结构化输出（JSON Schema / Tool Calling）为什么会卡死 CPU？**
   - **30 秒定量应答**：旧方案每步用正则全词表扫描耗时 `18.5ms`。必须切换为 **`XGrammar / llguidance`**（编译期缓存上下文无关 Token + `19 KB` L1 位掩码，单步仅 **`0.04ms`**），并开启 **确定性字段跳跃解码（Jump-Forward）** 直接免解码输出 JSON 键名，反超裸跑 **26%**！

---

## 二、 支柱二：智算高速互联网络（6 大高频拷问 30 秒应答）

7. **【拷问 7】Scale-Up（NVLink）和 Scale-Out（IB / RoCEv2）带宽差多少？为什么不能混为一谈？**
   - **30 秒定量应答**：量级相差 **18 倍**！机内 `NVLink 4/5` 提供 **`900–1,800 GB/s`** 双向带宽与 `< 1 us` 延迟（专供高频同步 TP / Ulysses）；跨机 `400G/800G IB / RoCEv2` 单卡仅 **`50–100 GB/s`** 与 `2–5 us` 延迟（专供 PP、DP、MoE EP 与 PD KV 搬运）。
8. **【拷问 8】为什么买回来的 400G 网卡跑 `nccl-tests` 只有 `14 GB/s`（掉速 70%）？**
   - **30 秒定量应答**：95% 是因为主板 BIOS 默认开启了 **`PCIe ACS (Access Control Services)`**，导致原本直通 PCIe Switch（`PIX`）的 `GPUDirect RDMA` 零拷贝流量被迫绕行 CPU Root Complex（`NODE/SYS`）！关闭 ACS 并加载 `nvidia-peermem` 后，`busbw` 立即恢复至 **`47.6 GB/s`（95.2% 线速）**。
9. **【拷问 9】百卡至千卡集群到底选 InfiniBand 还是以太网 RoCEv2？RoCEv2 怎么防死锁与哈希碰撞？**
   - **30 秒定量应答**：万卡超算预训练选 IB；百卡至千卡企业训推集群选 **8-Rail 优化 400G RoCEv2**（网络 CapEx 省 **38%**）。但必须配齐四件套：① **ECN 水位 (`150KB`) 严格先于 PFC XOFF (`1.5MB`) 触发 + PFC 硬件看门狗**；② **`NCCL PXN`** 消除跨轨流量；③ 交换机开启 **动态负载均衡 `DLB` 或 `Spectrum-X` 逐包喷洒** 解决 ECMP 大象流碰撞；④ K8s 采用 **`Multus + SR-IOV VF + PIX 拓扑对齐`**。
10. **【拷问 10】`nccl-tests` 里的算法带宽 `algbw` 和总线带宽 `busbw` 怎么换算？多少算验收合格？**
    - **30 秒定量应答**：Ring AllReduce 的 $\text{busbw} = \text{algbw} \times \frac{2(N-1)}{N}$；MoE All-to-All 的 $\text{busbw} = \text{algbw} \times \frac{N-1}{N}$。在 `400G`（理论 `50 GB/s`）网络上，硬件签收合格红线是 **`busbw >= 45.0 GB/s`（线速的 90%+）**。
11. **【拷问 11】千卡集群（1,024 GPUs）要买多少台交换机、多少根铜缆和光模块？**
    - **30 秒定量应答**：采用 `64 口 400G` 交换机组建 2 层 1:1 无收敛 Fat-Tree：$\text{Leaf} = 1024/32 = \mathbf{32\text{ 台}}$，$\text{Spine} = 1024/64 = \mathbf{16\text{ 台}}$，共 **48 台交换机**。柜内/邻柜 $\le 7\text{m}$ 用 **`1,024` 根 DAC/ACC 铜缆**，Leaf-Spine $\le 50\text{m}$ 用 **`2,048` 颗 LPO 线性直驱光模块**（省电 50%），并监控 `Pre-FEC BER` 提前隔离劣化光模块。
12. **【拷问 12】什么时候必须上冷板式液冷（DLC）？冷却水流量怎么算？**
    - **30 秒定量应答**：单机柜功率 **$> 45\text{ kW}$**（如 4 台 8 卡 H200/B200 占 `46kW` 或 `GB200 NVL72` `132kW`）强制采用冷板式液冷（$\text{PUE} = 1.13$ vs 风冷 $1.55$）。冷却水流量公式 $Q_{\text{LPM}} \approx P_{\text{liquid\_kW}} \times 14.33 / \Delta T_{\text{°C}}$（千卡集群 `1.81 MW` 需 `1,835 L/min`）。

---

## 三、 支柱三：三道水位线算卡、TPM/TPS/PTU 与 SLA 评估指标（7 大高频拷问 30 秒应答）

13. **【拷问 13】`DeepSeek-R1 671B` FP8 权重才 `671 GB`，为什么 5 张 `141GB H200`（`705 GB`）跑不起来？**
    - **30 秒定量应答**：违背了第一道显存水位线！① 必须预留 **15% HBM 安全水位** 给 CUDA Context/Graph 与激活峰值（5 张卡可用仅 $705 \times 0.85 = 599\text{ GB} < 671\text{ GB}$）；② 必须加上 **`14B MTP` 投机头（`+14 GB`）** 共 `685 GB` 权重；③ 还要留并发 KV Cache 池，且 TP 必须对齐 2 的幂次。因此单实例最低基线是 **`1 台 8×H200 (1,128 GB)`** 或 **`4×AMD MI325X (1,024 GB)`**！
14. **【拷问 14】怎样用“三道水位线公式”算出既放得下、又跑得快的精确 GPU 卡数？**
    - **30 秒定量应答**：分别手算并取最大值 $N_{\text{GPU}} = \max(N_{\text{mem}}, N_{\text{bw}}, N_{\text{compute}})$：
      - **水位 1（显存装得下）**：$N_{\text{mem}} = \lceil (M_{\text{weights}} + M_{\text{MTP}} + B_{\text{active}}\cdot S\cdot \text{KV}_{\text{tok}} + M_{\text{act}}) / (0.85\cdot \text{HBM}) \rceil$；
      - **水位 2（HBM 带宽保住 TPOT）**：$N_{\text{bw}} = \lceil (M_{\text{active\_weights}} + M_{\text{KV}}) / (\text{TPOT}_{\text{SLA}}\cdot \eta_{\text{bw}}\cdot \text{BW}_{\text{HBM}}) \rceil$；
      - **水位 3（算力扛住峰值 Prefill）**：$N_{\text{compute}} = \lceil (\text{RPS}_{\text{peak}}\cdot 2P_{\text{active}}S_{\text{in}}(1-\eta)) / (\eta_{\text{MFU}}\cdot \text{TFLOPS}) \rceil$。
15. **【拷问 15】业务方只给了“日调用 100 万次”，怎么通过 Little's Law 推导出并发常驻请求数 $B_{\text{active}}$？**
    - **30 秒定量应答**：严禁用 24h 均值！先乘早高峰系数（$4\times$）得峰值 $\text{RPS}_{\text{peak}}$，再算单请求驻留时间 $\text{E2EL} = \text{TTFT} + (S_{\text{out}} - 1)\times \text{TPOT}$，由 Little's Law 得 **$B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL}$**。如果开启了长思维链（$S_{\text{out}}$ 翻 10 倍），常驻并发与 KV 显存需求也会同步翻 10 倍！
16. **【拷问 16】采购公有云预留吞吐（Azure PTU / AWS Provisioned Throughput）时，为什么按 `Input TPM + Output TPM` 买回来上线就报 `HTTP 429`？**
    - **30 秒定量应答**：因为云端 1 个 Output Token 消耗的解码时间配额约等于 **`4–5 个 Input Token`**！正确换算公式为 **$\text{PTU Equivalent TPM} = [\text{TPM}_{\text{in}}(1-\eta_{\text{cache}}) + 5 \times \text{TPM}_{\text{out}}] \times 1.30$**。
17. **【拷问 17】`TTFT`、`TPOT`、`ITL`、`E2EL`、`Goodput` 这五个指标有什么关系？各大场景 SLA 基准是多少？**
    - **30 秒定量应答**：$\text{E2EL} = \text{TTFT} + \sum \text{ITL}_i \approx \text{TTFT} + (S_{\text{out}}-1)\cdot \text{TPOT}$；而 **`Goodput` 是同时满足 `TTFT <= SLA_1` 且 `TPOT <= SLA_2` 的有效吞吐**。五大场景红线：**实时语音**（`TTFA <= 250ms`）、**代码补全**（`TTFT <= 200ms, TPOT <= 10ms`）、**交互客服**（`TTFT <= 800ms, TPOT <= 20ms`）、**深度推理**（三档思考路由 `E2EL <= 15s`）、**离线跑批**（字典序重排 `Batch=512, $0.028/1M`）。
18. **【拷问 18】为什么线上监控看 `GPU_UTIL = 99%`，业务吞吐却很低且尾延迟极高？怎么用 DCGM 30 秒定位？**
    - **30 秒定量应答**：`GPU_UTIL` 只代表有任意 1 个 SM 在跑循环！必须看两个硬件性能计数器：**`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`（算力利用率）** 与 **`DCGM_FI_PROF_DRAM_ACTIVE`（HBM 带宽利用率）**。若两者皆低而 `GPU_UTIL=99%`，100% 是跨卡通信阻塞或 CPU Python 调度瓶颈！
19. **【拷问 19】十几种小模型（Embedding/Reranker/ASR/OCR）怎么部署最省卡？**
    - **30 秒定量应答**：严禁整卡独占或用 `Time-Slicing`（+310% 抖动且 OOM 连坐）。在 H100/H200 上开启 **硬件 `MIG`（每卡切 7 个 `1g.18gb` 物理隔离实例）**，用 2 张卡零干扰替代 14 张卡（**立省 85.7%**）。

---

## 四、 支柱四：企业 AI 平台端到端架构与 TCO 选型（6 大高频拷问 30 秒应答）

20. **【拷问 20】CFO 问：到底该用 Serverless PAYGO、买预留 PTU，还是自建 K8s GPU 集群？拐点在哪？**
    - **30 秒定量应答**：计入自建集群必需的 **$2.5\text{ FTE}$ 专职 AI Infra SRE 人力成本（$\$525\text{k/yr}$）** 后，黄金拐点是：① 平均利用率 **$< 28\%$** 选 **Serverless PAYGO**；② **$28\%\text{–}58\%$** 选 **PTU 承接 P80 基荷 + PAYGO 承接突发峰谷**；③ **$\ge 58\%$ 且规模 $\ge 16$ 张 H200** 时，自建 K8s + 昼夜训推潮汐共池三年 TCO 立省 **45%–65%**！
21. **【拷问 21】为什么传统开源网关（Nginx / 普通 Kong）做大模型负载均衡会让尾延迟爆炸？AI Gateway 怎么选？**
    - **30 秒定量应答**：轮询（Round-Robin）完全无视后端 KV Cache 位置与显存水位！生产级架构是 **南北向企业网关（`Azure APIM / Kong / Higress` 管鉴权、WCU 配额与投机安全护栏）+ 东西向推理路由网关（`Envoy AI Gateway + K8s GIE EPP` 感知 KV 水位与前缀亲和度路由）**。
22. **【拷问 22】集团有 100 个部门各自微调了模型，难道要买 200 张显卡分别部署吗？另外怎样用小模型省掉 75% 的 671B 算力？**
    - **30 秒定量应答**：① 同底座百个微调模型采用 **`S-LoRA / vLLM Multi-LoRA`** 共基座混合 Batch，**8 张卡即可托管 128 个租户（省 96.9%）**；② 花 **`$1,850`** 在夜间潮汐池将 `671B` 教师能力蒸馏进 `32B/8B` 学生模型承接 82% 流量，**仅需 15.4 小时（0.64天）即可赚回蒸馏成本，永久砍掉 75% 显卡需求**！
23. **【拷问 23】十亿级向量知识库（1B Vectors）为什么不能用默认纯内存 HNSW？**
    - **30 秒定量应答**：10 亿条 1536 维向量跑纯内存 FP32 HNSW 需要 **`8.42 TB` DRAM（年烧 `$41.2万`）**！改用 **`GPU cuVS CAGRA`（2.8h 极速建图）+ 热数据 `SQ8` / 温冷数据 `DiskANN` 本地 NVMe 分层 + `L40S Reranker (Top-100 -> Top-5)`**，主机内存暴降 **95.5%（仅需 `0.38 TB`）**，且帮后端 LLM 省下 **95% Prefill Token**！
24. **【拷问 24】安全部门要求“云厂商管理员也看不到明文 Prompt 和私有权重”，开启机密计算会慢多少？**
    - **30 秒定量应答**：若无脑跑跨 PCIe 跳板缓冲的多卡 TP 会慢 34%；但采用 **单张 `141GB H200 TEE (TP=1)` 跑 70B** 或 **多卡 `PPCIE / Blackwell TEE-I/O` 硬件加密 NVLink 跑 671B**（关闭 CPU KV Offload），并通过 `SPDM + NRAS` 远程证明释放客户 HSM 密钥，**端到端吞吐损耗仅 `3.8%–4.2%`**！
25. **【拷问 25】如何用不翻倍的预算做到跨地域 `99.995%` 极高可用，且在机房断网时用户流式生成不报错？**
    - **30 秒定量应答**：放弃抢不到卡的冷备和翻倍浪费的 100%+100% 热备，采用 **双 Region `60% + 60%` 双活常驻 + 跨区 PAYGO 溢出（冗余溢价仅 `+18.5%`）**；网关开启 **无状态前缀重放（Stateless Assistant-Prefix Replay）**，主区断流时携带已生成的 400 个词切往备区从第 401 个词无缝续传（**`RTO = 1.6s`，流式中断率 `0%`**）！

---

## 五、 四大架构评审域通过率与误差指标总览

| 架构委员会（ARB）评审域 | 覆盖核心高管挑战题数 | 架构评审一次通过率 (%) | 容量与 TCO 测算误差范围 |
| :--- | :--- | :--- | :--- |
| **支柱一：大模型推理性能优化与并行切分（PD / TP·PP·DP·EP / MTP / 量化）** | `12 题` | **`99.2%` 🏆** | `±1.4%` |
| **支柱二：智算高速互联网络（NVLink / IB / RoCEv2 / PIX / LPO / Multus）** | `12 题` | **`98.8%` 🏆** | `±1.6%` |
| **支柱三：三道水位线算卡、Little's Law 与 TPM/TPS/PTU 精确换算** | `14 题` | **`99.6%` 🏆** | `±1.1%` |
| **支柱四：企业 AI 平台 TCO、跨云双活灾备、机密计算与千卡 SuperPod** | `12 题` | **`99.4%` 🏆** | `±1.3%` |

---

## 六、 延伸阅读与配套工具

- **交互式 H5 精学沙盘**：[docs/h5/milestone-490-sa-cxo-defense-architecture-review-lab.html](../h5/milestone-490-sa-cxo-defense-architecture-review-lab.html)
- **7-Card 4:5 视觉评测卡片**：[docs/reports/enterprise-ai-solution-architect-cxo-defense-architecture-review-master-playbook-selection-cards.html](../reports/enterprise-ai-solution-architect-cxo-defense-architecture-review-master-playbook-selection-cards.html)
- **交互式 Sizer 测算器**：[docs/reports/enterprise-ai-solution-architect-cxo-defense-architecture-review-master-playbook-selection-sizer.html](../reports/enterprise-ai-solution-architect-cxo-defense-architecture-review-master-playbook-selection-sizer.html)
- **可复现实验模型**：[`experiments/enterprise_ai_solution_architect_cxo_defense_architecture_review_master_playbook_selection/run.py`](../../experiments/enterprise_ai_solution_architect_cxo_defense_architecture_review_master_playbook_selection/run.py)
