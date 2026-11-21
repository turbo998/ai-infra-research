# 推理性能、智算网络与容量测算：架构师深度 FAQ

**语言**：简体中文 | [English](faq.en.md) · [H5 精学馆](../h5-academy/index.html) · [双语课程索引](README.md)

> **证据合同**：本文是企业 Cloud & AI Solution Architect 学习指南，不是硬件 benchmark。公式算例与场景目标均为 `synthetic` 教学假设；官方规格与实现依据见[来源目录](../h5-academy/sources.json)，复核于 2026-11-21。不存在独立复现实机的性能承诺。

## Executive Summary：先体验，再容量，后拆分与网络

1. **用户侧**同时看首字等待、流式平滑度、完成时间与质量；机器侧看达标 Goodput，不只看无限制峰值 TPS。
2. **业务侧**用峰值 QPS 和输入/输出 Token 分布求需求；Little's Law 需要平均停留时间，不能拿 P99 当等式。
3. **GPU 侧**分别跨过显存、Decode 读带宽和 Prefill 算力三道水位线；671B MoE 的 37B 激活不能替代总权重。
4. **架构侧**先验证混部 + Chunked Prefill，再比较 PD 的净收益；没有固定的“16 卡必须 PD”阈值。
5. **网络侧**统一 GB/GiB、Gb/s/GB/s、单向/双向、单端口/聚合与 NCCL 消息口径，再讨论瓶颈。

```mermaid
flowchart TD
    A["业务画像：峰值 QPS、输入/输出分布、质量"] --> B["体验合同：TTFT、ITL、E2EL、Goodput"]
    B --> C["吞吐与并发：TPS/TPM、L=λ×E[W]"]
    C --> D["三道水位线：显存 / 读带宽 / Prefill 算力"]
    D --> E{"混部 + Chunked 达标吗？"}
    E -- "是" --> F["保持简单共池，验证冷热缓存与突发"]
    E -- "否，且 trace 显示 P/D 干扰" --> G["算 KV 交接、两池排队与复制成本"]
    G --> H["配置 TP/PP/DP/EP 组与 NVLink/RDMA 域"]
    H --> I["真实 payload 与拥塞测试，验证达标成本"]
```

## 专题 1：用户 AI 核心评估指标

### Q1. TTFT、TPOT/ITL 和 E2EL 怎么区分？

TTFT 是请求发出到首 Token 可见的时间，包含客户端/网关、排队、Prefill、首字计算和交付。服务端 TTFT 与端到端 TTFT 要分别命名。

对于输出 `N` 枚 Token：

```text
TPOT = (最后 Token 时间 − 首 Token 时间) / (N−1)
ITL_i = 第 i 枚 Token 时间 − 第 i−1 枚 Token 时间
E2EL = TTFT + (N−1)×TPOT
```

TPOT 是单请求后续间隔的平均值；P99 ITL 能揭示短时卡顿。输出仅一枚 Token 时 TPOT 为 **N/A**，不是 0。网络 chunk 可能包含多枚 Token；不要用 chunk 数冒充 tokenizer Token 数。

实际采集有约定差别：vLLM 的逐请求 TPOT Prometheus histogram 可将 `N_out≤1` 记成0，而 `vllm bench serve` 可排除这些样本；其 ITL 还可能按 stream output event 而非逐 Token 计。数学 N/A、采集零值和 benchmark 过滤必须区分，不能用零值推出无限 TPS。单靠两个独立 histogram 不能精确按请求关联输出长度，应保留关联 trace。

### Q2. 可以把各阶段 P99 相加当作整体 P99 吗？

不能。分位数不线性可加，且请求阶段彼此相关。保留同一请求 trace，先对每个请求求总时延，再取整体分位数。相加的阶段预算可以是工程预算，但不能宣称是观测到的 E2EL P99。

### Q3. 五种业务场景该优先看什么？

| 场景 | 教学目标，不是通用 SLA | 评估重点 |
|---|---|---|
| 实时语音 | 端到端 turn response `<300 ms`；示例给 LLM TTFT 120 ms | VAD/ASR/网络/LLM/TTS 预算；首音频不等于首文本 |
| IDE 补全 | TTFT 200 ms、TPOT 15 ms、64 输出 Token | 短请求、取消过时结果、低排队 |
| 企业对话 | TTFT 800 ms、TPOT 35 ms、512 输出 Token | RAG 输入长度、冷热 prefix、流式尾延迟 |
| R1 深度推理 | TTFT 2 s、TPOT 50 ms、4096 输出 Token | 质量、总完成时间、首可见答案与隐藏推理 Token |
| 离线批处理 | 小时级完成截止时间；交互 TTFT 为 N/A | 成本、总完成时间、失败/重试、达标吞吐 |

语音手算预算示例：VAD 20 + ASR 40 + LLM 120 + TTS 60 + 网络/网关 30 = **270 ms**。这是拆预算练习，不保证真实系统在 270 ms 完成。离线场景不画成“TTFT=0”。

### Q4. Throughput 和 Goodput 为什么会背离？

Throughput 必须先声明计数群体：完成请求率、生成 Token 速率、客户端收到的 Token 速率并不是同一口径；失败前已发出的 Token 可以计入原始流量，但不能伪装成成功完成的请求。Goodput 只计同时满足约定 SLA 和质量门槛的部分。高 batch 可能增加 TPS，却使 TTFT 或 ITL 超标；此时 Goodput 下降甚至为零。

H5 的 tradeoff 曲线来自可调确定性教学模型，不是硬件帕累托前沿。等输出长度下可用请求达标比例乘 Token TPS；不等输出长度要按实际 Token 加权。

新增数值时间戳解剖台：7条教学请求、18枚可见 Token，保留失败、拒绝、品质未过与单 Token。burst 的可见间隔 `0,0,60 ms` 平均是20 ms，按平均 TPOT 35 ms 协议能过，按最大 ITL 35 ms 不能过；同一个原始18 tok/s，达标输出会从 **9降到5 tok/s**。这些是合成记录，不外推生产 P99；重复 chunk 时间不代表 GPU 每 Token 0 ms。

并发直接由请求区间面积求得。默认 client 边界下 `L=1.475`，所有区间完整落在1秒测窗内，`λ×平均W` 恰好相等是有限追踪恒等式，不是稳态能力证明。把窗口截到200 ms，面积 `L=4.695`，而窗内终态率×终态平均停留时间仅 **1.345**；边界截断不能当扩容结论。admitted 边界也可能包含服务端排队，不等于 GPU busy。

新性能采集可关注 **NVIDIA AIPerf**：截至2026-11-21，GenAI-Perf 官方已标明新功能开发退场并推荐 AIPerf。学习页不安装工具、不接推理端点、不引用不同机器的示例数字证明自己的性能。

**生命周期算例**：排队 120 ms + Prefill/首字 300 ms + 交付 30 ms = TTFT 450 ms。128 输出 Token、TPOT 20 ms，则 E2EL = 450 + 127×20 = **2990 ms**，后续流式约 **50 tok/s**。

## 专题 2：TPM/TPS 与企业容量

### Q5. 单用户 TPS、System TPS 与 TPM 的公式一样吗？

```text
单用户后续流式 TPS ≈ 1000 / TPOT_ms
System Input TPS = 稳态完成 QPS × 平均输入 Token
System Output TPS = 稳态完成 QPS × 平均输出 Token（含首 Token）
同口径 TPM = 60×TPS
```

10 QPS、每请求输入 2048、输出 512，则需求为 **20,480 Input TPS** 和 **5120 Output TPS**。单用户 50 tok/s 不等于整个系统只能处理 50 tok/s。过载时 offered QPS 不等于 completed QPS。

### Q6. Little's Law 的“并发”究竟是什么？

`L=λ×E[W]`：L 为平均在系统内的请求数，λ 为稳定通过系统的请求率，W 为同一系统边界的平均停留时间。它既不是 DAU、在线人数，也不是固定 batch。

按 10 QPS，加 30% 需求余量后 λ=13。规划平均 TTFT 800 ms、TPOT 20 ms、512 输出 Token，则：

```text
W = 0.8 + 511×0.02 = 11.02 s
L = 13×11.02 = 143.26
Input TPM = 13×2048×60 = 1,597,440
Output TPM = 13×512×60 = 399,360
```

这里把目标延迟当成**规划均值假设**，不冒充已证实均值。观测验证必须换成实测 E[W]，突发、排队增长与拒绝需要另外压测。

混合负载不能先把输出长度和TPOT分别平均后相乘。教学 IDE/RAG/推理三类的请求占比为50%/40%/10%，停留分别1.145/18.685/206.75秒：平均W为 **28.7215秒**，计划13 QPS对应L=373.3795。推理请求只占10%，却占约 **71.98%平均在途**；错误均值乘积只算到17.7231秒。每类平均参数仍需同类实测，L不是自动等于GPU驻留量。

短时突发也不等于“结束即恢复”：服务10 RPS、基线8 QPS，30秒3倍突发积压 **420请求等价量**；恢复余量2 RPS，需要 **210秒**排空。300ms仅排队预算约能容3个流体请求等价量，多余约417需拒绝/转移。这是固定平均μ的流体近似，不预测真实P99，也不创造额外服务容量。

### Q7. 怎么从吞吐与并发求实例数？

使用**同一混合业务负载**测得每实例输入/输出吞吐以及活跃请求上限：

```text
实例下界 = max(
  ceil(Input TPS / 可用输入容量),
  ceil(Output TPS / 可用输出容量),
  ceil(L / 活跃请求上限)
)
```

教学实例容量：输入 20K TPS、输出 1500 TPS、活跃上限 128；使用 70% 实测吞吐容量。上一例分别得到输入 2、输出 7、并发 2 个实例，先以 **7 个实例**压测。一个实例可以由多张 TP/PP/EP GPU 构成，不是一张卡。

### Q8. PTU 能直接换算 TPM 或 GPU 吗？成本如何算？

没有通用换算。按当前具体模型、版本、部署类型、缓存与上下文权重归一化，再按最小值与扩容步长取整。两权重教学示例：

```text
Normalized TPM = Input TPM×[(1−cache)+cache×cached_weight]
               + Output TPM×output_weight
```

Microsoft Learn 的 GPT-5.2 Data Zone 算例：200K 输入 TPM、20K 输出 TPM、输出权重 8、每 PTU 3400 输入等价 TPM。50% 缓存且缓存权重 0 → **260K normalized TPM → 76.47 PTU → 80 PTU**。H5 最小值为手工演示字段；采购前核对真实部署表。GPT-6+、cache writes、图像和上下文档位可能需要更多权重，不能默认当免费。

截至2026-11-21，GPT-6.1/6.2归一化快照明确将每个Token只归入一次短/长普通输入、缓存读、30分钟写、12小时写或输出。短/长缓存读权重0.05/0.1、普通输入1/2、输出5/7.5、30分钟写1.25/2.5、12小时写2.5/5；GPT-6 Sol/Luna/Astra读权重为0.1/0.2。短/长档位依具体官方模型/定价定义，不假设统一128K阈值。

H5默认分类264K原始TPM→ **271K归一化TPM**。`gpt-6.2-sol /2026-11-17`预算2950 normalized TPM/PTU，raw91.86，Global/Data Zone按最小15/步长5取 **95 PTU**，Regional按25/25取 **100 PTU**。错误把缓存读当免费会得到90个Global PTU。不能重复计cache-write为ordinary input；这不是订阅quota、地区capacity或价格查询。零新需求也不会让已部署PTU闲置费用消失。

自建完整集群成本 `$40/h`、达标输出 `5000 tok/s`：

```text
$/1M 达标输出 = 40×10⁶/(3600×5000) = $2.22
```

API 输入/输出价格分别算，不与自建单输出分母混用。Goodput=0 时单位成本未定义，不显示 `$0`。

**云额度与吞吐还差一道“限流计量器”**：2026-11-21直读 Microsoft Learn 的新/经典 quota 页面，两者均说明使用实际输入与生成 Token，而非把配置的最大输出值预留进 TPM。旧检索摘要仍描述 `max_tokens` / `best_of` 估算，应以当前原文为准；这里不宣称服务变更日期或具体部署已实测。RPM 可按1或10秒短窗口判断：官方600 RPM示例若用1秒窗，超过10请求/秒就可能429，哪怕整分钟少于600。核对部署的 remaining/reset、`retry-after-ms` 与实际 usage；PTU 另用利用率评估。两份现行文档一致，文档依据高置信；上线前仍须复核当前文档和部署行为。见[新文档](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/quota#understanding-rate-limits)及[经典文档](https://learn.microsoft.com/azure/foundry-classic/openai/how-to/quota#understanding-rate-limits)。

## 专题 3：GPU 三道水位线

### Q9. 为什么 Dense 与 MoE 必须区分总参数与激活参数？

GPU 常驻的全模型主权重用总参数算。DeepSeek-V3/原始 R1 主模型约 671B，单 Token 激活约 37B；37B 不会让其他专家权重消失。额外加载 MTP 模块需再预算 **14B**。CPU/NVMe offload 属于另一种部署边界，必须计搬运和 CPU 计算。

```text
M_weight_GB = (total_params_B + optional_MTP_B)×weight_bytes×(1+metadata_fraction)
```

BF16/FP16 2 B、FP8 1 B、INT4 0.5 B 是原始存储近似。量化 scales、padding、未量化层、runtime 和额外状态要单独预算。

### Q10. MHA、GQA 与 MLA 的 KV 公式怎么用？

```text
MHA/GQA KV bytes = 2×L×n_KV×d_head×S×B×KV_bytes
MLA compressed KV bytes = L×(latent_rank+RoPE_dim)×S×B×KV_bytes
```

Llama 70B GQA：80 层、8 KV heads、128 head dim，BF16 每 Token **327,680 B**。S=8192、B=32 → **85.89934592 GB=80 GiB**。

DeepSeek 压缩 MLA：61 层、latent 512、RoPE 64，BF16 每 Token **70,272 B**；均匀 FP8 假设为 **35,136 B**。后者要求实际 kernel 对所有分量使用 1 B，不能默认认为每种实现都如此。TP 复制、额外 MTP state 与 scales 会改占用。

### Q11. 70B 在 H200 上“放得下”的最低卡数怎么手算？

FP8 70.6B，5% 元数据余量 → 权重 74.13 GB；加上一例 KV 85.90 GB。每卡安全水位 85%、运行时预留 3 GB：

```text
每卡安全 payload = 141×0.85−3 = 116.85 GB
理想显存下界 = ceil((74.13+85.90)/116.85) = 2 GPU
```

这是理想聚合分片下界，不是完整 SLA 推荐。KV 与权重未必按理想比例分片；最后看最忙 rank 的实际峰值。

### Q12. 为什么显存 2 卡下界，沙盘候选却可能是 8 卡？

读带宽与算力是另两道约束：

```text
TPOT_weight_floor ≈ active_weight_GB/(N_GPU×BW_GBps×MBU)
带宽场景还要加每步 KV 读取
Prefill FLOPs ≈ 2×active_params×S_p×B_prefill
              +2×L×hidden×S_p²×B_prefill
```

同一默认算例使用 H200 4.8 TB/s、MBU 0.6、目标 TPOT 20 ms，权重+KV 读带宽场景下界为 **3 卡**。Prefill 使用 dense BF16 989.5 TFLOPS、MFU 0.4、TTFT 800 ms 减预留 100 ms，1 个完整 8K Prefill 的算力下界为 **5 卡**。取三者最大值，再给待压测的 2 的幂候选 **8 卡**，仍不是采购保证或已验证 TP 配置。

冷 Prefill 输入 `S_p` 与峰值 KV 驻留 `S` 必须分开：8192 输入加最多512输出，应设 `S_p=8192`、`S≈8704`。只增加未来输出应增加 KV，不应虚增这一次冷 Prefill FLOPs。`B` 包括同实例所有 KV 驻留请求，`B_prefill≤B`；32 个 Decode 加一个新 Prefill 不能仍只给32个 KV 请求位。

进一步看每 rank：vLLM LlamaAttention 中 `KV_heads/rank=max(1,n_KV/TP)`，且切分必须可整除。70B 的8个 KV heads 在 TP=8/16 时均是每 rank 1 head，因此 **TP=16 的总 KV 为 TP=8 的2倍**，不能继续理想地除以16。MLA 参考布局可能保留完整 latent/RoPE；opaque packed 格式须录入真实 row bytes，而非照 head_size×dtype 猜。

新增每 rank 沙盘用普通 `TP×PP×副本DP`、PP层数、权重偏斜预算、最忙副本活动数和设备总可见显存逐一验证。普通 DP 复制整模型，不是 Attention DP+EP。实测字段需你主动填入已确认的数据，默认仍为示例；不要把加载后剩余 free 显存当总容量再扣一次权重。PP内存能省，不表示单请求 Decode 的串行路径能获得 `N×HBM` 加速。

MoE 大 batch 的专家并集会接近更多总权重。H5 的均匀独立路由模型用于解释，不能把单 Token 37B 流量直接用于整个 batch。

规格需纠偏：L40S 48 GB 为 **GDDR6**，864 GB/s，**没有 NVLink**；H100/H200 为 SXM 指定规格；B200 **192 GB design/what-if** 与官方 DGX B200 **180 GB/卡**分开。80–85% 是规划余量，不是宣称 vLLM 当前默认。

## 专题 4：PD 分离

### Q13. 混部干扰的根源和 Chunked Prefill 的上限是什么？

Prefill 常更吃算力，低 batch Decode 常更吃内存带宽；二者同时争 SM、kernel 时序、缓存、显存与通信。Chunking 减少单段 Prefill 阻塞，但资源仍共用。更小 chunk 可能改善 ITL，却拉长长输入等待；没有通用最优 chunk。

### Q14. 什么时候值得进入 PD 验证？

trace 证实 Prefill 干扰 Decode，chunking 后双 SLA 仍失败，且 KV 交接+两池排队+计算能落入 TTFT 预算，并有持续网络、故障与复制成本余量。没有通用“16 卡以上必须 PD”规则。

低 QPS/小 KV 的 TCP PD 也可能成立；“没有 RDMA 一律禁止”是过度绝对化。RDMA/NVLink 对大 KV 低延迟场景更有价值，但仍需版本、拓扑与真实 payload 验证。

### Q15. KV 跨节点传输如何手算？

70B FP8 KV、8192 输入 → **1,342,177,280 B**。400 Gb/s 单向为 50 GB/s，80% 效率为 40 GB/s：

```text
Raw transfer = 1,342,177,280/(40×10⁹)×1000 = 33.554432 ms
Exposed handoff = max(0, raw−verified_overlap) + setup
```

已证实重叠 10 ms、固定 0.2 ms → **23.754432 ms**。10 QPS 的持续 payload 占该方向预算 **33.55%**。若到达量超过有效带宽，持续队列不稳定；不能继续套一个固定传输时间宣称 TTFT 达标。

逻辑KV、实际跨网字节和D侧引用状态是不同量。只有实际网络方案要求多份发送时才乘wire倍率；D侧通过NVLink本地广播复制，不必把跨网量同步翻倍。D已有前缀减少新传输，却仍占引用KV；峰值状态还应计未来输出。opaque KV packing/scales可覆盖理论每Token字节，但必须录入唯一未重复的逻辑尺寸。

PD预算沙盘沿用同一传输参数，取P输入、D输出、D驻留槽和网络容量的最小值。默认教学上界P200K输入TPS、D12K输出TPS、256 D槽、10QPS时，D输出约束 **23.4375请求/s**，候选系统输出上界12K tok/s；不是实测Goodput。每rank内存、模型/Tokenizer/dtype/block/layout与同负载时延还须独立确认。

首个可见Token来自哪里也影响合同：D交接后首字预算约 **573.75 ms**；P先发首字可约 **500 ms**，但首次后续间隔约 **93.75 ms**，仍不满足35 ms流式预算。这里相加的是工程预算，不是各阶段P99。只有后台真实支持并被trace证实的layer overlap才可扣除；部分pull路径在P完成后才开始传。

### Q16. Mooncake、Dynamo、vLLM、SGLang 是互斥替代品吗？

不是。Dynamo 是编排/路由/弹性层；Mooncake 提供 KV-centric 架构与 transfer/store 生态；vLLM/SGLang 是推理引擎与相应 connector/runtime。部署可能组合这些层。真正核对后端 release、KV layout/dtype、TP reshaping、缓存正确性、故障恢复和两池容量。

H5每点生成 **200个32输出Token请求**，每请求31个确定性后续间隔，共 **6200个ITL样本**。先求每请求平均TPOT，再对200个请求取P99；同批6200间隔另外取P99 ITL，两者不能混名。普通请求约每20间隔一次干扰，2%请求额外突发。假设首字已从D可见，不含启动交接，也不预测网络过载；PD仍保留残留抖动，不承诺“无抖动”。

2026-11-21的GitHub release元数据快照为Dynamo **v1.6.2**、vLLM **v0.34.0**、SGLang **v0.5.23**、Mooncake **v0.3.14**。这不是已验证联合栈；主线/索引文档可能晚于release。部分混合attention recipe要求P/D同TP或限制MTP+PD，NIXL pull/push对PCP/DCP支持也不同。机内有NVLink不保证跨Pod走CUDA IPC，仍须核对PID namespace和真实传输日志。

## 专题 5：并行和推理加速

### Q17. TP、PP、DP、EP 到底切什么？

| 方式 | 切分维度 | 通信与主要代价 |
|---|---|---|
| TP | 同一层张量/权重 | AllReduce 或 RS+AG；高频同步 |
| PP | 层/流水 stage | SendRecv activations；bubble 与串行路径 |
| 普通推理 DP | 独立请求副本 | 路由、权重与缓存复制；没有训练梯度归约 |
| Attention DP + EP | Attention 请求分片与共享专家组 | 组语义依引擎；不是所有专家全复制 |
| EP | MoE 专家 | AllToAll dispatch/combine；热点与 token 交换 |

### Q18. 怎么决定 NVLink 域内与跨节点切分？

高频 TP 优先快域，但不要把节点名当作快域边界。NVSwitch/MNNVL 超节点可以扩展域。PP/DP 常跨域，EP 需要高质量 AllToAll。记录 GPU↔NIC 亲和、实际并行组、oversubscription 与通信重叠；不存在普遍适用的绝对禁令。

### Q19. 为什么投机、Prefix Caching、FP8 不该承诺固定倍数？

投机收益依接受率、draft/验证成本、batch 饱和、采样与约束模式；额外 draft/MTP 占显存。Prefix Caching 只复用相同 tokenized 前缀，主要节省 Prefill，不直接把 Decode 翻倍。FP8/INT4 改变存储/算子，但质量、硬件、scale、反量化和 kernel 决定净收益。

投机手算：接受率独立同分布、α=0.8、每轮 draft γ=4，则期望提交 `1+0.8+0.64+0.512+0.4096=3.3616 Token`，不是每轮都 5 Token。基线 target 20 ms、每候选 draft 2 ms、每候选增加验证 1 ms、提交固定 1 ms，则 cycle=33 ms，条件性平均服务时间约 **9.82 ms/Token，2.04×**。这是公式算例，不能替代实际接受率、验证成本、质量或 P99 ITL；多 Token 突发提交还可能掩盖周期停顿。

前缀手算：8192 输入中有 4096 可共享 Token，70% 请求命中只平均避免 `0.7×4096=2867.2 Token`，即 **35% 输入计算**。20K Prefill TPS 与 2 ms 查找开销下，平均 Prefill 从 409.6 ms 变成 **268.24 ms**；缓存全部 miss 时反而多出 2 ms。10 个独立 70B BF16 前缀条目要额外预算约 **13.42 GB KV**，共享 block 和复制/驱逐策略另算。

### Q20. PP microbatch 越多越好么？

理想平衡流水线利用率 `m/(m+p−1)` 随 microbatch 增多而提高，但真实系统还有不等长请求、慢 stage、KV 驻留、队列与通信。单请求仍走所有 stages；高系统吞吐不自动等于低 TTFT。

4 stages、8 microbatches 的理想 forward-only 时序共 `8+4−1=11` 个 stage 时间槽，利用率 **8/11=72.73%**。第一条请求仍走 4 个 stage，新增 batch 不会把这条串行路径缩短。H5 可看到同一 microbatch 沿各 stage 对角线移动，也可切换 TP 列切/行切、DP 独立副本与 EP top-2 玩具 dispatch/combine；玩具 top-2 不冒充 DeepSeek 的实际 top-8。

## 专题 6：RDMA、RoCEv2、NVLink 与 NCCL

### Q21. GPUDirect RDMA “零拷贝”是不是完全没有 CPU？

它让 payload 避免 host DRAM staging，NIC 直接 DMA 到注册 GPU 内存。CPU/驱动仍负责连接、注册、页固定、队列、控制与错误处理。检查 GPU/NIC PCIe 亲和、IOMMU/ACS、peer-memory/DMA-BUF 与驱动支持。零 host-copy 不等于零开销。

数据仍经过实际PCIe端点/交换路径。Gen3/4/5单向编码后带宽为 `GT/s×lanes×128/130÷8`：Gen5 x16约 **63.02 GB/s**，再按85%payload假设约 **53.56 GB/s**；800G端口单向线速100 GB/s也可能被PCIe卡住。活动端口数不等于NIC卡PCIe链路数，双端口卡要共享相应端点预算。Gen6/7 PAM4/FLIT不能套这个编码公式。

默认节点8 GPU端点、8张400G单端口卡，在80%网络效率下320 GB/s；共享路径300、分配fabric240，上界 **240 GB/s**。1024 MB完全条带下界4.27 ms；只有1个GPU端点时可能只剩53.56 GB/s。GPU型号不自动决定实际协商的PCIe代际/lanes，预算也不是模型吞吐保证。

同节点也不保证跨Pod走NVLink。Dynamo索引文档中的UCX CUDA IPC需要PID namespace/peer GPU可见性等条件；不满足时可能转RDMA或TCP且不报NIXL错误。检查日志和实际payload，不把privileged/hostPID当默认修复，多租户隔离要经安全认可。

### Q22. PFC、ECN 和 DCQCN 分别做什么？

传统 lossless RoCE 基线：交换机 ECN 提前标 CE → 接收 RNIC 发 CNP → 发送 RNIC DCQCN 降速/恢复。若继续逼近溢出，PFC 在单跳单 priority 上暂停上游，队列低于 Xon 后恢复。

ECN 是早反馈；PFC 是逐跳急刹，不是端到端拥塞控制。要防 PFC 死锁/HOL，验证 watchdog、计数器与 incast。现代 fabric 也可能采用不同拥塞控制、receiver credits 或 loss-tolerant RDMA；本指南不是唯一组合的配置命令。

headroom需吸收pause生效前继续进入的流量。工程下界约 `link_Bps×reaction_seconds+max_frame+guard`：400G、10µs、9KB帧、50KB额外余量约 **0.559 MB**；16MB教学buffer在80% Xoff后留3.2MB。800G、100µs时要求约 **10.059 MB**，同样80%阈值已不够。百分比不是厂商profile，真实ASIC/线卡共享buffer限制另核。

### Q23. 900/1800 GB/s NVLink 与 400/800G 网络怎么比较？

先统一口径。NVLink 4/5 的 900/1800 GB/s 为每 GPU 聚合双向宣传规格，非一个端口或单流独享。400/800 Gb/s 单端口单向理论分别是 **50/100 GB/s**，还需扣协议、效率与争用。NDR 400G / XDR 800G 需核对 lane、port、cable 实际组合。

Rail-Optimized 把对应 GPU/NIC 索引映射到轨，提高亲和与路径可控性；跨轨仍依赖 spine/路由，不能自动承诺“轨数×线速”。

玩具映射沙盘中，8 GPU各发1GB到4条400G rail、80%效率，均衡负载2/2/2/2GB，下界 **50ms**；全部碰撞一rail则 **200ms**。丢一rail并允许重映射后，3/3/2GB需 **75ms**，不是必然3×40GB/s满载。不开重映射时完整传输不能完成，保留阻塞字节而非伪装成功吞吐。热点与fabric预算可另成瓶颈；不模拟真实routing/recovery。

### Q24. NCCL algbw 与 busbw 为什么不同？

```text
algbw = S/t
AllReduce busbw = algbw×2(n−1)/n
AllGather / ReduceScatter / AllToAll busbw = algbw×(n−1)/n
Broadcast / Reduce / SendRecv: factor=1
```

1000 MB/20 ms → algbw **50 GB/s**。8 ranks AllReduce → busbw **87.5 GB/s**。这是通信量标准化，不是单端口凭空超线速。AllGather/ReduceScatter 的 S 使用总数组，不是每 rank 单块。对比必须同操作、消息大小定义、路径、topology、dtype、warmup 与错误状态。

若输入的是NCCL API `count×dtype_bytes`，AllGather(sendcount)、ReduceScatter(recvcount)、固定AllToAll(per-peer count)还需乘n才是总S；AllReduce不乘。8 ranks、每API块1000 MB、20ms的AllGather对应S8000 MB、algbw400、busbw350 GB/s，不是50/43.75。使用实际对齐后的count/报告size；AllToAllv要看最大send/recv，不按平均专家流量套固定公式。

## 决策矩阵：先找到哪一种失败，再选手段

| 观测失败 | 优先排查 | 可能动作 | 必须复验 |
|---|---|---|---|
| TTFT 高、decode 平滑 | 排队、长输入、冷热 prefix、Prefill 算力 | 路由、缓存、输入治理、算力扩容 | 同分布 TTFT、质量、成本 |
| ITL 尾部被长 Prefill 拉高 | 请求关联 trace、chunk budget | Chunked Prefill，必要时 PD | TTFT+ITL 双门槛、KV 交接 |
| GPU OOM | 总权重、KV、每 rank 复制/碎片/图捕获 | 合法 TP/PP/EP、精度/KV、限制并发 | 最忙 rank 峰值、质量 |
| 大 batch MoE 吞吐差 | 专家并集/热点、AllToAll、attention | 调并行组、亲和、专家负载均衡 | 热点、通信、Goodput |
| 网络队列/PFC 风暴 | ECN/CNP、映射、incast、拥塞 | 调流控/拓扑/接入，验证 watchdog | 单轨故障、真实 EP+KV 共载 |
| 成本很低但用户超时 | 分母是否用了失败吞吐、闲置是否忽略 | 用达标输出重算全成本 | 明确 SLA、时间窗与质量 |

## 上线前核对清单

- [ ] 同一模型/version/tokenizer、输入输出分布、质量与采样合同。
- [ ] 服务端 TTFT、端到端 TTFT、首音频、首可见答案分别命名。
- [ ] Little's Law 使用平均 W；突发/过载/拒绝另外验证。
- [ ] 权重总参数、KV 表示/dtype、MTP、每 rank runtime 与实际显存字节。
- [ ] GPU form factor、dense/sparse FLOPs、GB/GiB、单向/双向口径。
- [ ] 混部 chunked 基线与 PD 原始/重叠/暴露交接分别测量。
- [ ] 并行组、GPU/NIC 亲和、专家热点、KV layout 与故障恢复。
- [ ] 网络小消息 latency、大消息带宽、incast、PFC/ECN/CNP 与轨故障。
- [ ] 用达标 Goodput/质量评估成本，不用无限制峰值。

**复算入口**：[H5](../h5-academy/index.html) · [来源](../h5-academy/sources.json) · [测试内核](../h5-academy/academy-math.cjs)。
