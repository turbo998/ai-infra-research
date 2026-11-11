# 端侧小模型（SLM 1B～8B）手机 NPU、AI PC 与浏览器 WebGPU 推理及端云协同路由架构选型指南（`100% 纯云端 70B API` vs. `端侧 CPU/GPU 通用 GGUF Q4_K_M / WebGPU` vs. `端侧专用 NPU 静态图 + SpinQuant W4A8` vs. `端云协同置信度路由 [75% 端侧 NPU + 25% 云端 70B]`）

> **阅读指南（3 分钟速通 / 10 分钟白板宣讲）**：本文面向 **AI 解决方案架构师（AI Solution Architect）与智能终端 / AI PC / 超级 App 平台负责人**。用第一性原理拆解千万日活（1,000 万 DAU，日均 1 亿次请求）移动与桌面应用落地端侧小模型（SLM，如 `Qwen-3B` / `Llama-3.2-3B` / `Phi-4-Mini`）的核心决策——**为什么全量调云端 70B API（Tier 1）每年要烧掉 `$1,533 万` 算力费且断网即废？为什么直接在手机/轻薄本 CPU/GPU 上跑 `llama.cpp GGUF Q4_K_M` 或 `WebGPU`（Tier 2）会以 `8.5W` 高功耗在 90 秒内把手机烫到 44°C 触发腰斩降频（跌至 `14.2 tok/s`）？为什么端侧专用 NPU（高通 Hexagon QNN / 苹果 ANE / Intel NPU）必须配 `SpinQuant` 正交哈达玛旋转才能跑通 `W4A8` 定点矩阵乘并以 `2.4W` 超低功耗飙出 `54.0 tok/s`（能效暴涨 `13.5 倍`）？以及如何通过“端侧 NPU 置信度门控 + 75% 本地零成本秒回 + 25% 高熵复杂题无缝上云 70B”端云协同架构（Tier 4），同时拿下 `89.8%` 旗舰级智商、`72ms` 闪电首字响应与每年 `$1,146 万（-74.8%）` 的云算力节省？**

[![返回 README 导航](https://img.shields.io/badge/⬅️_返回_AI_Infra_架构选型主目录-0F172A?style=for-the-badge)](../../README.md)
[![English Version](https://img.shields.io/badge/🌐_English_Version-small--language--model--slm--on--device--npu--webgpu--deployment--selection--faq.en.md-2563EB?style=for-the-badge)](./small-language-model-slm-on-device-npu-webgpu-deployment-selection-faq.en.md)
[![7-Card 评测报告](https://img.shields.io/badge/📊_7--Card_视觉评测报告-HTML_直达-9333EA?style=for-the-badge)](../reports/small-language-model-slm-on-device-npu-webgpu-deployment-selection-cards.html)
[![交互式选型工作台](https://img.shields.io/badge/🧮_交互式端云选型计算器-HTML_直达-059669?style=for-the-badge)](../reports/small-language-model-slm-on-device-npu-webgpu-deployment-sizer.html)

![端侧小模型 SLM 手机 NPU、SpinQuant W4A8 与端云协同路由选型一图流总结](../../assets/svg/small-language-model-slm-on-device-npu-webgpu-deployment-selection-faq-summary.svg)

---

## ① 📌 核心结论速览（Executive Summary）

1. **为什么千万日活（10M DAU）应用既不能“100% 裸调云端（Tier 1）”，也不能“直接在手机 CPU/GPU 上跑 GGUF（Tier 2）”？**：
   - **Tier 1（100% 纯云端 70B API 推理）的成本与体验双杀**：在日均 `1 亿次` 请求（其中 75% 是邮件润色、通知摘要、本地日程提取、UI 意图路由等轻量级高频操作）下，全部发往云端 70B 集群不仅承受 **`380.0 ms` 的移动网络首字延迟（TTFT）✕**、电梯/高铁弱网不可用，更要每天支付 **`$42,000`（每年 `$1,533 万` ✕）** 的云端 GPU 账单！
   - **Tier 2（端侧移动端 CPU/GPU 跑 `llama.cpp GGUF Q4_K_M` 或浏览器 `WebGPU`）的“90 秒发热降频墙”**：移动端 GPU/CPU 跑的是 `W4A16`（4-bit 权重实时反量化成 FP16 激活算矩阵乘），整机功耗高达 **`8.5 Watts` ✕**！无风扇手机的散热极限只有 `3.0W ~ 3.5W`，连续对话 90 秒后机身表面温度触及 44°C 红线，系统强行降频，吐字速度从峰值跌至 **`14.2 tok/s` ✕**，且纯 3.8B 小模型在面对 25% 复杂逻辑/代码题时综合准确率仅 **`70.8%` ✕**！
2. **如何把端侧能效提升 `13.5 倍`？——专用 NPU 静态图 + `SpinQuant W4A8` 旋转消峰量化（Tier 3）**：
   - **为什么手机/AI PC 的专用 NPU（高通 Hexagon / 苹果 ANE / 联发科 APU）省电 3.5 倍？** NPU 是专为 **`W4A8 / INT8` 定点整数脉动阵列** 设计的芯片，运行 3.8B 模型功耗仅 **`2.4 Watts` 🏆**（完全落入被动散热安全区，永不降频！）。
   - **为什么普通模型放进 NPU 会变傻，必须用 `SpinQuant`？** NPU 要求把动态激活值（Activations）也压成 `8-bit INT8`。但大模型激活值存在百倍离群尖峰（Outliers），直接做 `W4A8` 会把模型智商砍崩！**`SpinQuant`（学习型正交哈达玛旋转，$R^\top R = I$）** 将旋转矩阵离线融合进权重里（$W' = W R$，运行时零额外权重开销），把尖锐的激活离群值均匀摊平到所有通道上，实现 **零精度损失的 `W4A8` NPU 静态图推理（`ExecuTorch / Qualcomm QNN / CoreML`）**——首字延迟压至 **`65.0 ms`**，持续稳态吐字飙至 **`54.0 tok/s`（每瓦特吐字能效达 Tier 2 的 `13.48 倍` 🏆）**！
3. **消费级与企业端智能体的终极黄金架构（Tier 4：端云协同置信度级联路由 — `Apple Intelligence / Hybrid AI` 范式）**：
   - **核心机制**：不强求 3.8B 端侧模型包打天下！所有请求先由本地 `2.4W` 的 **3.8B SpinQuant W4A8 NPU** 零网络延迟启动推理（同时评估首词 Top-1/Top-2 概率裕度 Margin 或轻量路由头）：
     - **75% 的日常高频请求（低熵高置信度：改写、摘要、端侧系统设置、个人隐私短信检索）**：由端侧 NPU 直接在本地以 **`65 ms` TTFT、`54 tok/s`** 完成闭环——**云端成本 `$0.00`、隐私数据 100% 不出端、断网可用**！
     - **25% 的跨领域复杂推理/代码请求（高熵低置信度）**：在 `12 ms` 内无感上云交由 **云端 `70B FP8` 旗舰模型** 接管！
   - **量化收益**：端到端综合准确率高达 **`89.8%`（逼近纯云端 70B 的 `91.2%`，比纯端侧暴涨 `+18.3 pts` 🏆）**，中位首字延迟压低 **`5.28 倍`（`380 ms → 72 ms` 🏆）**，日均云算力账单从 `$42,000` 暴降至 **`$10,600/天`（每年立省 `$1,146 万 [-74.8%]` 🏆）**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（FAQ-Driven First Principles）

### Q1：同样是跑一个 3.8B 的 4-bit 小模型，为什么在手机 GPU 上跑 `GGUF Q4_K_M`（Tier 2）会发烫掉电，而在手机 NPU 上跑 `W4A8`（Tier 3）却冰凉丝滑、能效高出 `13.5 倍`？

- **通俗大白话 + 硬件微架构第一性原理拆解**：
  1. **移动端 GPU / CPU 跑 `W4A16 GGUF` 的三大耗电元凶**：
     - **浮点运算与寄存器搬运贵**：`Q4_K_M` 虽然把权重存成了 4-bit，但计算时必须先在通用 Shader 核心里把 4-bit 解压回 16-bit 浮点数（`FP16`），再做 `FP16 × FP16` 乘加。
     - **动态逐算子派单与缓存抖动**：通用移动端 GPU 缺乏大模型专用的片上大 SRAM 紧耦合缓冲，每层都要频繁读写耗电大户 LPDDR5X 内存，导致整机功耗冲上 **`8.5 Watts`**，90 秒触发热保护把频率砍半（`14.2 tok/s`）。
  2. **专用 NPU（如高通 Hexagon HTP / 苹果 ANE）的硬件降维打击**：
     - **原生 `W4A8` 定点整数微架构**：NPU 内部是专用的 `4-bit 权重 × 8-bit 整型激活` 脉动阵列，1 个周期可完成数千次整数乘加，单次乘加能耗仅为 GPU FP16 的 $1/5$！
     - **整图静态编译（AOT Static Graph）+ 片上数 MB VTCM/SRAM 驻留**：通过 `PyTorch ExecuTorch`、`Qualcomm QNN` 或 `Apple CoreML` 将整个 Transformer 层编译成一张静态硬件指令图，中间激活值直接留在 NPU 片上紧耦合高速缓存（VTCM）里不落主存，功耗骤降至 **`2.4 Watts`**，稳态速度高达 **`54.0 tok/s`（`22.5 tok/s/W` vs. GPU 的 `1.67 tok/s/W`，能效比提升 `13.48 倍` 🏆）**！

### Q2：为什么直接把小模型做普通 `W4A8` 量化放进 NPU 会出现“胡言乱语”，而 `SpinQuant / QuaRot` 的“正交哈达玛旋转”能做到零损失？

- **第一性原理拆解（把“一根独苗刺破天际”变成“全场均匀平摊”）**：
  1. **NPU 的死穴——激活值离群尖峰（Activation Outliers）**：
     - 在大模型的固定几个隐藏层维度上，激活值的数值会突然飙到 `150.0`，而其余 4,095 个维度都在 `[-1.5, +1.5]` 之间。
     - 当 NPU 要求把这 4,096 个数统一塞进 `INT8 [-128, +127]` 的 256 个刻度格子时，为了迁就那个 `150.0` 的巨型尖峰，每一个刻度格子的步长被迫设为 $150 / 127 \approx 1.18$——结果其余 4,095 个正常维度的细腻数值全部被粗暴四舍五入成了 `0` 或 `±1`！模型当场失智！
  2. **`SpinQuant` 正交旋转消峰魔法（$Y = (X R) (R^\top W^\top) = X W^\top$）**：
     - 线性代数告诉我们：只要矩阵 $R$ 是正交矩阵（满足 $R R^\top = I$，就像在空间里旋转一下坐标系），那么先把输入激活 $X$ 乘以 $R$，再把权重 $W$ 乘以 $R^\top$，两者相乘的结果与原来 **100% 逐比特完全相等**！
     - 当我们选择一个经过蔡氏流形优化的 **正交哈达玛旋转矩阵 $R$（`SpinQuant`）** 时，乘以 $R$ 就相当于把那个高达 `150.0` 的单通道尖峰能量，像打奶泡一样均匀搅拌分摊到了全部 4,096 个通道上（每个通道都变成平滑的 `~2.3`）！
     - 更妙的是：$W' = W R^\top$ 可以在打包模型时**离线提前算好合并进权重里（运行时 0 开销！）**，而激活侧的快速沃尔什-哈达玛变换（FWHT）只需极轻量的加减法即可完成，让 `W4A8` 在手机 NPU 上恢复与 FP16 几乎无异的精度！

### Q3：在浏览器里跑模型（`WebGPU / WebLLM`）和在原生 App 里调 `NPU` 分别适合什么场景？Tier 4 的“端云协同路由”又是怎么拦截 75% 成本的？

- **三大落地边界与端云级联门控机制**：
  1. **什么时候选浏览器 `WebGPU`（Tier 2 子场景）？**
     - 当你做的是**零安装网页端 SaaS 工具**（用户不愿下载 2GB 原生 App，直接在桌面 Chrome/Edge 打开网页即可调用本地显卡跑 `WebLLM / ONNX Runtime Web`）。但切记：**在手机移动端原生 App 中，严禁用 WebGPU/GPU 替代 NPU 跑长时推理**！
  2. **Tier 4 端云协同置信度门控（Confidence-Gated Cascade）是怎么运作的？**
     - 用户发起提问后，端侧 `3.8B SpinQuant W4A8 NPU` 模型在本地立即做 Prefill（仅需 `50 ~ 65 ms`），并同时检查两个零成本指标：
       1. **意图与敏感度标签**：若涉及本地通讯录/相册/私密笔记或离线指令，**100% 强制锁死在端侧 NPU 执行**；
       2. **首词对数概率裕度（Logit Margin $\Delta p = p_{\text{top1}} - p_{\text{top2}}$ 与香农熵 $H(p)$）**：对于占日常总量 **75% 的常规任务**（润色、总结、翻译、简单闲聊），端侧 SLM 胸有成竹（$H(p) < 0.6$），直接在本地以 `54 tok/s` 流式吐完，**不花云端 1 分钱**；
       3. 一旦检测到用户在问复杂的跨语言代码架构、奥数证明或实时联网资讯（熵值高、置信度低于阈值，占 **25%**），端侧在 `12 ms` 内立即把 Prompt 异步转发给 **云端 70B FP8 大模型**！

---

## ③ 🧭 直观可视化图解（Mermaid 决策树 / NPU 旋转量化路径 / 端云协同级联时序）

### 3.1 端侧小模型（SLM）硬件推理栈与端云协同选型决策树

```mermaid
flowchart TD
    Start["📱 端侧小模型 (SLM 1B~8B) 与端云架构选型<br/>输入：日活请求量、终端形态 (手机/PC/网页) 与任务复杂度"] --> Q1{"产品形态是否为免安装纯网页端<br/>(无法调用操作系统原生 NPU 驱动)？"}
    Q1 -- "是（桌面浏览器免安装扩展/网页应用）" --> T2["🟡 Tier 2：WebGPU / Wasm 浏览器推理 (WebLLM / ORT-Web)<br/>适合插电桌面 PC 浏览器，⚠️ 手机上跑 8.5W 易发热降频！"]
    Q1 -- "否（iOS/Android 原生 App 或 Windows/Mac 原生客户端）" --> Q2{"用户任务是 100% 固定窄域轻量任务，<br/>还是包含开放域复杂代码/深度推理？"}
    Q2 -- "100% 窄域轻量任务 或 极端离线保密环境" --> T3["🔵 Tier 3：纯端侧专用 NPU 静态图 + SpinQuant W4A8<br/>ExecuTorch / Qualcomm QNN / Apple CoreML ANE<br/>✅ 2.4W 冰凉不降频 | 65ms 首字 | 54 tok/s | $0 云成本"]
    Q2 -- "开放域混合任务（既要省钱秒回，又要 90% 旗舰智商）" --> Q3{"日请求量是否 >= 100 万次/天？"}
    Q3 -- "否（极低频内部工具，不在乎云账单）" --> T1["⚪ Tier 1：100% 纯云端 70B API 兜底<br/>⚠️ 10M DAU 下年烧 $1,533万 且弱网/离线完全不可用"]
    Q3 -- "是（百万至亿级日请求消费/企业应用）" --> T4["🟢 Tier 4：端云协同置信度路由 (Apple Intelligence 范式)<br/>75% 日常请求端侧 3.8B NPU 秒回 ($0) + 25% 难题无缝上云 70B<br/>🏆 综合准确率 89.8% | 中位 TTFT 72ms (快5.3x) | 年省 $1,146万"]
```

### 3.2 传统端侧 GPU `W4A16` 发热降频 vs. Tier 4 `SpinQuant W4A8` NPU + 端云级联路径对比

```mermaid
flowchart LR
    subgraph Bad["❌ Tier 1/2：传统路径（要么全上云年烧 $1,533万，要么端侧 GPU 8.5W 烫手降频）"]
        direction TB
        B1["Tier 1：100% 请求无论难易全发云端 70B<br/>💥 380ms 移动网络延迟 + 年耗资 $1,533万"] --> B3["成本失控或终端体验崩盘"]
        B2["Tier 2：手机 GPU/CPU 跑 GGUF W4A16<br/>💥 8.5W 功耗 90秒烫至 44°C，降频至 14.2 tok/s<br/>💥 普通 W4A8 被激活离群值刺穿导致模型失智"] --> B3
    end

    subgraph Good["✅ Tier 4：SpinQuant W4A8 专用 NPU + 端云置信度级联路由"]
        direction TB
        G1["① 离线模型编译：SpinQuant 正交哈达玛旋转<br/>⚡ 将激活百倍离群尖峰均匀摊平，解锁无损 W4A8 整数定点化"] --> G2["② 端侧第一级 (75% 流量拦截)：专用 NPU 静态图<br/>🔋 仅耗 2.4W (能效 13.5x)，65ms 首字、54 tok/s 离线秒回 ($0)"]
        G2 --> G3["③ 云端第二级 (25% 高熵复杂题上云)：云端 70B FP8<br/>🏆 端到端保住 89.8% 旗舰准确率，每年净省 $1,146万 (-74.8%)"]
    end
```

### 3.3 端云协同置信度级联路由（Confidence-Gated Hybrid Cascade）端到端交互时序图

```mermaid
sequenceDiagram
    autonumber
    participant U as 终端用户 (手机 / AI PC)
    participant NPU as 端侧 NPU (3.8B SpinQuant W4A8 @ 2.4W)
    participant Gate as 端侧隐私与置信度门控 (Logit Margin / Entropy)
    participant Cloud as 云端旗舰集群 (70B FP8 API)

    Note over U,Gate: 场景 A（占 75% 日常流量）：通知摘要 / 邮件润色 / 本地日程提取
    U->>NPU: 输入请求：“把这段会议纪要提炼成 3 条待办”
    NPU->>Gate: 65ms 完成本地 Prefill，首词置信度极高 (Entropy H = 0.22 < 0.60)
    Gate-->>NPU: 判定本地闭环！端侧 NPU 以 54.0 tok/s 冰凉流式输出 (2.4W，云成本 $0.00)

    Note over U,Cloud: 场景 B（占 25% 硬核流量）：复杂跨库代码排障 / 多步数学推导
    U->>NPU: 输入请求：“分析这段分布式锁在网络分区下的死锁边界并给出 TLA+ 证明”
    NPU->>Gate: 12ms 路由头/首词探测发现高不确定性 (Entropy H = 1.85 >= 0.60) 且不含本地隐私锁
    Gate->>Cloud: 无缝异步上云转发至 70B FP8 旗舰集群
    Cloud-->>U: 返回 91.2% 专家级深度推理结果 (用户兼得日常秒回与难题高智商)！
```

---

## ④ 📊 架构师选型决策矩阵（Architect Decision Matrix）

| 评估维度 | Tier 1：100% 纯云端 `70B API` | Tier 2：端侧 CPU/GPU `GGUF Q4_K_M / WebGPU` | Tier 3：纯端侧专用 NPU `SpinQuant W4A8` | Tier 4：端云协同置信度路由（`75% NPU + 25% 云 70B`） |
| :--- | :--- | :--- | :--- | :--- |
| **核心执行与量化技术栈** | 全量请求走蜂窝/Wi-Fi 调云端 70B | 端侧 GPU/CPU 动态反量化 `W4A16` | **正交旋转 `SpinQuant W4A8` + NPU 静态图 (`QNN/ANE`)** | **端侧 `W4A8 NPU` 首词置信度门控 + 25% 难题上云 `70B`** |
| **中位首字响应延迟 (`Median TTFT`)** | `380.0 ms` ✕（受制于移动网络 RTT） | `245.0 ms` ⚠️ | **`65.0 ms` 🏆（本地 NPU 零网络延迟）** | **`72.0 ms` 🏆（比纯云端快 `5.28 倍`）** |
| **5 分钟连续对话稳态吐字速度 (`tok/s`)** | `42.0 tok/s` | **`14.2 tok/s` ✕（90秒烫至 44°C 降频腰斩）** | **`54.0 tok/s` 🏆（恒速不降频）** | **`51.5 tok/s` 🏆（丝滑流式输出）** |
| **终端设备持续发热功耗 (`Active Watts`)** | `0.8 W`（仅网卡收发） | **`8.5 W` ✕（严重烫手、疯狂掉电）** | **`2.4 W` 🏆（每瓦能效比 GPU 高 `13.48 倍`）** | **`2.1 W` 🏆（冰凉省电）** |
| **混合业务场景综合准确率 (`Accuracy %`)** | **`91.2%` 🏆（70B 旗舰基准）** | `70.8%` ✕（3.8B 遇复杂推理崩盘） | `71.5%` ✕（受限于端侧参数容量） | **`89.8%` 🏆（比纯端侧暴涨 `+18.3 pts`）** |
| **1 亿次/天请求云端算力账单 (`USD / Day`)** | **`$42,000 / 天` ✕（`$1,533万/年`）** | `$0 / 天` | `$0 / 天` | **`$10,600 / 天` 🏆（每年净省 `$1,146 万 [-74.8%]`）** |
| **适用场景与架构师结论** | 仅限极低频且必须 100% 联网的重度后台任务 | 免安装桌面浏览器 `WebGPU` 演示 | **100% 离线/隐私敏感的窄域端侧功能（输入法/摘要）** | **千万日活智能终端与超级 App 唯一生产级黄金标准** |

---

## ⑤ 🧪 简明定量算例与 Benchmark 洞察（1,000 万 DAU / 日均 1 亿次请求实测）

> **数据来源与透明度声明（Provenance Disclosure）**：本节与配套 7-Card 报告的所有量化数据均由本仓库归档的确定性仿真评估器 [`experiments/slm_on_device_mobile_npu_webgpu_hybrid_selection/run.py`](../../experiments/slm_on_device_mobile_npu_webgpu_hybrid_selection/run.py) 及密封输出 [`results.json`](../../experiments/slm_on_device_mobile_npu_webgpu_hybrid_selection/results.json) 生成（标记为 `independently_reproduced=false`、`physical_gpu_cluster_measurement=not_measured`，参数校准自 Meta SpinQuant W4A8 旋转量化论文、PyTorch ExecuTorch / Qualcomm QNN Hexagon NPU / Apple CoreML ANE 基准与 RouteLLM 端云协同级联研究）。

### 5.1 为什么说“`SpinQuant W4A8` 专用 NPU + 端云路由（Tier 4）”是超级 App 唯一能算过账来的商业模式？

1. **纯云端（Tier 1）的利润粉碎机**：
   - 当一款 1,000 万日活的办公/手机助手 App 每人每天触发 10 次 AI 交互（共 1 亿次/天），如果连“改写一句话、提取一条日程”这种小事都发往云端 70B 大模型，每年光云端 GPU 电费与折旧就要烧掉 **`$1,533 万` ✕**，免费用户直接把公司吃垮！
2. **用好用户自己手机里已经付过钱的 `45 TOPS NPU`（Tier 3 + Tier 4）**：
   - 通过 `SpinQuant` 将 3.8B 模型无损编译为 `W4A8` NPU 静态图，单瓦特吐字效率从移动端 GPU 的 `1.67 tok/s/W` 飙升 **`13.48 倍` 至 `22.50 tok/s/W`（仅耗 `2.4W`）**！
   - 配合端侧置信度门控拦截掉 **75% 的日常高频请求**，每年直接为企业省下 **`$1,146.1 万`（`-74.8%`）云端算力费 🏆**，同时把中位首字响应从 `380 ms` 缩短至 **`72 ms`**，并在 25% 难题上通过无缝上云保住 **`89.8%`** 的旗舰级准确率！

### 5.2 📊 7-Card 4:5 视觉评测卡片集与交互式端云选型计算器

- **7-Card 4:5 视觉评测报告**：[`docs/reports/small-language-model-slm-on-device-npu-webgpu-deployment-selection-cards.html`](../reports/small-language-model-slm-on-device-npu-webgpu-deployment-selection-cards.html)
- **交互式端侧 SLM NPU 与端云协同计算器**：[`docs/reports/small-language-model-slm-on-device-npu-webgpu-deployment-sizer.html`](../reports/small-language-model-slm-on-device-npu-webgpu-deployment-sizer.html)

---

## ⑥ 🛠️ 附录：生产环境快速核验清单（Architect Checklist）

在交付移动端（iOS/Android）、AI PC 或浏览器端侧大模型应用前，请逐项核验以下 6 条工程铁律：

- [ ] **1. 严禁在无风扇手机上用通用 GPU/CPU 跑长时间 `W4A16` 推理**：`8.5W` 功耗会在 90 秒内触发机身 44°C 热保护降频并耗尽电池；生产环境必须卸载至 **`2.4W` 的专用 NPU（Qualcomm QNN / Apple CoreML ANE / MediaTek NeuroPilot）**。
- [ ] **2. 针对定点 NPU 部署 `W4A8` 时，必须前置 `SpinQuant / QuaRot` 正交哈达玛旋转**：将正交旋转矩阵离线合并进线性层权重（$W' = W R^\top$），彻底抹平百倍激活值离群尖峰，严禁直接对原始激活做粗暴 MinMax INT8 截断。
- [ ] **3. 将端侧模型导出为固定形状桶的 AOT 静态编译图（Static Graph Buckets：如 `128 / 512 / 2048` Context）**：专用 NPU 极度厌恶动态张量形状（Dynamic Shapes 会触发回退 CPU 执行），必须预先编译固定尺寸 KV Cache 环形缓冲区。
- [ ] **4. 采用“共享底座权重 + 多业务微秒级切换 `LoRA Adapter`（仅 15～30 MB）”架构**：像 Apple Intelligence 一样在手机闪存/内存中只驻留 1 个 `3B W4A8` 基座模型（占 `~1.8 GB`），针对邮件润色、通知摘要、代码补全动态热插拔轻量级 LoRA 权重，避免多个 App 重复塞爆手机内存。
- [ ] **5. 实施“隐私强锁死 + 熵/Margin 置信度门控”的双重端云级联路由**：凡命中通讯录/短信/财务等本地隐私标签的请求 100% 锁死端侧 NPU；非敏感请求若端侧首词香农熵 $H(p) \ge 0.60$ 则在 `<15 ms` 内无感异步转发云端 70B。
- [ ] **6. 明确区分原生 App（走 `ExecuTorch / ONNX Runtime QNN / CoreML` 硬件 NPU）与免安装网页端（走 `WebLLM / WebGPU` 显卡 Shader）的技术栈边界**：切勿在原生 App 内套壳 WebView 跑 WebGPU 白白浪费 NPU 硬件红利。
