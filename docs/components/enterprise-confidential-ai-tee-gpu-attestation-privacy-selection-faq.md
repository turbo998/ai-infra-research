# 企业级高机密数据隐私与机密计算（Confidential AI）大模型架构选型指南：标准云端明文内存 GPU vs. 本地物理隔离机房（Air-Gapped On-Prem）vs. 云端机密虚拟机 + GPU 硬件 TEE（H100/H200 CC）vs. 零信任双端远程证明（Remote Attestation）+ HSM 安全密钥释放（SKR）架构

> **文档类型**：企业级安全合规、主权云与 AI 解决方案架构师（Cloud & AI Solution Architect）选型决策指南（FAQ）  
> **适用读者**：银行/保险/医疗 CISO（首席信息安全官）、半导体与生物医药核心 IP 负责人、ToB 垂类模型独立软件开发商（ISV）、云原生 AI 架构师  
> **核心目标**：用通俗、可量化的架构语言回答高监管行业（金融、医疗、政务、芯片研发）与商业模型厂商最关心的终极安全命题——**“即使配了专线（Private Link）和零数据留存（ZDR），如果云厂商底层宿主机管理员或黑客直接对物理内存/显存做 Memory Dump，我们的机密 Prompt 和核心微调模型权重（Data-in-Use）是不是就全泄密了？”、“为了防内存窃取，难道只能花几千万、等 9 个月自建本地物理隔离机房吗？”、“基于 NVIDIA H100/H200 硬件机密计算（Confidential Computing）+ 远程证明 + HSM 密钥释放（SKR）是怎么做到仅用 5.8% 的性能代价实现 100% 密码学内存隔离、并比自建机房省下 55.2% 成本的？”**。  
> **标定声明**：本文所引用的 70B 推理吞吐损耗、上线交付周期、运行态数据保护得分与月度 TCO 模型基于公开发布的 GPU 机密计算基准（NVIDIA Hopper H100/H200 Confidential Computing 白皮书、Azure Confidential GPU VMs `NCCads H100 v5` 实测开销研究及机密容器 Kata CoCo 规范）构建的合成分析模型（`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`），详见 [`experiments/enterprise_confidential_ai_tee_gpu_attestation_selection/results.json`](../../experiments/enterprise_confidential_ai_tee_gpu_attestation_selection/results.json)。

---

## 一图流总结（Executive Visual Summary）

![企业级机密计算大模型 Confidential AI 架构选型总结](../../assets/svg/enterprise-confidential-ai-tee-gpu-attestation-privacy-selection-faq-summary.svg)

> 🔗 **关联可交互报告与工具**：
> - 🃏 **[7-Card 4:5 视觉评测报告 (`enterprise-confidential-ai-tee-gpu-attestation-privacy-selection-cards.html`)](../reports/enterprise-confidential-ai-tee-gpu-attestation-privacy-selection-cards.html)**
> - 🧮 **[交互式机密计算性能损耗与 TCO 计算器 (`enterprise-confidential-ai-tee-gpu-attestation-privacy-sizer.html`)](../reports/enterprise-confidential-ai-tee-gpu-attestation-privacy-sizer.html)**
> - 🌐 **[English Version (`enterprise-confidential-ai-tee-gpu-attestation-privacy-selection-faq.en.md`)](./enterprise-confidential-ai-tee-gpu-attestation-privacy-selection-faq.en.md)**

---

## ① 📌 核心选型结论速览（Executive Summary）

在传统云计算安全体系中，企业早已解决了 **静态数据加密（Data-at-Rest，磁盘 AES 加密）** 和 **传输数据加密（Data-in-Transit，TLS 1.3 专线加密）**；但唯一的致命软肋是 **运行态计算过程（Data-in-Use）**——当大模型推理时，你的机密 Prompt、检索出的核心代码/病历以及价值数百万美元的私有微调模型权重，都必须解密成**明文**躺在 CPU 内存（DRAM）和 GPU 显存（HBM）里！

基于 **70B-FP8 金融/医疗/核心 IP 大模型、峰值 100 RPS 并发** 的基准测算，四档部署架构对比如下：

| 架构档位 | 核心安全与基础设施组合 | 70B 推理性能损耗 | 上线交付周期 | 运行态内存与模型 IP 保护 | 月均总成本 (TCO) | 架构师拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟨 Tier 1：标准云端明文内存 GPU** | Private Link + 磁盘加密，但主机内存与显存运行态明文 | `0.0%` | `1 天` | `0.0%` ✕ (运行态明文) | `$24,000 / 月` | **常规业务可用**：但无法通过主权级“零云管理员信任”审计 |
| **🟥 Tier 2：100% 本地自建物理隔离机房** | 自购服务器 + 机房液冷/电力改造（Air-Gapped On-Prem） | `0.0%` | `270 天` (9个月) ✕ | `95.0%` (难防内部Root内鬼) | `$62,000 / 月` ✕ | **重资产死胡同**：周期长达 9 个月、成本高昂且防不住本机超管 |
| **🟦 Tier 3：云端机密 VM + GPU 硬件 CC** | **AMD SEV-SNP / Intel TDX + NVIDIA H100/H200 CC 模式** | `5.4%` ✅ | `2 天` ✅ | `92.0%` ✅ (硬件内存加密) | `$26,200 / 月` (-57.7%) | **运行态内存加密基线**：低成本堵上宿主机内存转储漏洞 |
| **🟩 Tier 4：零信任 TEE + 远程证明 + HSM SKR** | **Tier 3 + SPDM 双端远程证明 + HSM 安全密钥释放 (SKR)** | **`5.8%` 🏆** | **`2 天` 🏆 (快135x)** | **`100.0%` 🏆 (密码学零信任)** | **`$27,800 / 月` 🏆 (-55.2%)** | **主权金融/医疗与商业模型交付终局**：连云超管与本机Root都不可见 |

### 💡 架构师三句话选型铁律
1. **不要为了保护运行态数据（Data-in-Use）而盲目花几千万自建本地小机房（Tier 2）——直接用“CPU+GPU 硬件可信执行环境（Confidential AI TEE，Tier 3/4）”**：现代云端机密 GPU 实例（如 Azure `NCCads H100 v5` / GCP `a3-highgpu` Confidential）在芯片硅片级用硬件密钥加密 CPU 内存与 GPU HBM，**上线周期从 `9 个月` 缩至 `2 天`（快 `135 倍`），3 年期月均 TCO 立省 `55.2%`（`$27.8K` vs `$62.0K`）**！
2. **打破“加密会让 GPU 慢一半”的旧偏见——在 70B 大模型上开启 H100/H200 机密计算，吞吐损耗仅有 `5.4%~5.8%`！**：因为 GPU 机密计算只对跨越 PCIe 总线的数据包做硬件 `AES-GCM-256` 加解密；一旦 Token 进入 H100 显存内部，80 层 Transformer 矩阵乘法是 **100% 满血原生速度运行**的！模型参数越大、生成越长，PCIe 加密开销占比就越微不足道。
3. **保护高价值私有模型权重（Model IP）与客户极密数据的王牌组合是“加密容器镜像 + 双端硬件远程证明（Remote Attestation）+ HSM 安全密钥释放（SKR）”**：模型权重在磁盘和镜像仓库里永远是密文；只有当 CPU 和 GPU 芯片向客户自己掌控的硬件安全模块（HSM）提交不可伪造的硬件签名测量值（證明运行在真实未篡改的 TEE 飞地内且禁用了 SSH/Debug），HSM 才会把解密密钥直接打入飞地内存！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Architect Deep-Dive Q&A）

### Q1：我们已经签了云厂商的零数据留存协议（ZDR）并走了 Private Link，为什么央行/军工/半导体安全审计还是问“如果云平台底层宿主机 Hypervisor 被攻破或内部工程师拔内存条，怎么保证数据不泄露”？
* **真相剖析（从“合同承诺信任”升级到“硅片密码学零信任 Technical Zero-Trust”）**：
  - 传统的 ZDR 和 Private Link 属于**管理与网络边界控制**——在物理层面，云厂商宿主机（Host OS / Hypervisor）的 `root` 权限仍然高于你的普通虚拟机（Guest VM）。理论上，宿主机的 `dma_buf` 或内存转储工具仍然可以直接读取普通 VM 的物理内存页。
  - **Tier 4 硬件机密计算（Confidential Computing）怎么从物理定律上封死这条路？**
    1. **CPU 侧（AMD SEV-SNP / Intel TDX）**：CPU 内存控制器内嵌硬件 AES 引擎，密钥由芯片安全协处理器在开机瞬间随机生成并锁死在硅片内（任何软件可读接口都不存在）。宿主机 Hypervisor 如果去读该 VM 的内存地址，读出来的全部是高熵随机密文；如果试图篡改内存页，硬件完整性校验（SNP）会立即触发机器停机熔断！
    2. **GPU 侧（NVIDIA Hopper H100 / H200 Confidential Computing）**：GPU 内部硬件防火墙直接关闭所有 JTAG 调试端口、性能计数器侧信道和宿主机 BAR 显存直接映射；CPU 飞地与 GPU 飞地之间通过 **SPDM 协议建立端到端加密通道**，所有跨 PCIe 总线传输的 Prompt 和激活值全部经由 Bounce Buffer 用 **`AES-GCM-256`** 实时加解密！

### Q2：很多架构师听说早年 SGX 机密计算跑深度学习要慢 5~10 倍，为什么现在 H100/H200 跑 70B 大模型机密推理只慢 `5.8%`？
* **真相剖析（大模型的“高计算/通信比”天然契合 GPU TEE 架构）**：
  - 如果是小图像分类模型（如 ResNet-50），每次都要把几 MB 的未压缩图片跨 PCIe 传给 GPU，算 2 毫秒就结束了——这时候 PCIe 总线的 `AES-GCM` 加解密确实会带来 30%+ 的开销。
  - **但 70B 大模型推理完全相反！**
    - 140GB/70GB 的模型权重在容器启动时**一次性加载进 GPU HBM 显存后就常驻不动了**；
    - 每次用户提问，跨 PCIe 总线从 CPU 传给 GPU 的只有区区 **几个 KB 的输入 Token ID 整数数组**！
    - 随后 GPU 在自己受硬件保护的 HBM 显存内部独立完成长达几百毫秒、数万亿次浮点运算的 80 层 Attention 与 FFN 计算，最后只把生成的几个字节 Token ID 加密传回 CPU！
    - 因此，对于 `>= 13B~70B` 的大模型，或者开启了 Continuous Batching 的生产负载，PCIe 加解密耗时被庞大的 GPU 内部矩阵计算完全掩盖，实测端到端吞吐损耗仅为 **`5.4%~5.8%`**！

### Q3：我们是一家医疗/工业软件商（ISV），花了 500 万美金微调出了行业专有大模型，现在要部署到客户的私有云或公有云账号下，怎么防止客户的运维人员直接 `docker cp` 把我们的模型权重文件偷走？
* **真相剖析（Tier 4 机密容器 Kata CoCo + 远程证明 + HSM 密钥释放闭环）**：
  - 如果你把明文 `.safetensors` 权重交给客户机房（哪怕是本地 Air-Gapped Tier 2），客户方的 Linux `root` 管理员只需要一行命令就能把你的核心资产拷走！
  - **Tier 4 零信任模型 IP 保护流程**：
    1. **构建加密模型磁盘镜像（Encrypted Model Layer）**：发布前用对称密钥 $K_{\text{model}}$ 将 70B 权重文件全部加密，密钥 $K_{\text{model}}$ 保存在软件商自己控制的云端 HSM / KMS 中；
    2. **不可篡改的机密策略清单（Security Policy / Kata CoCo）**：容器启动策略硬编码规定 **`allow_exec = false`（彻底禁止 `docker exec / kubectl exec / SSH` 进入容器内部！）** 且禁止挂载外部未签名卷；
    3. **双芯片联合远程证明与安全密钥释放（Remote Attestation + SKR）**：只有当客户机器上的 CPU TEE（SEV-SNP/TDX）和 GPU TEE（H100 CC）同时通过远程证明服务校验——证明当前运行的代码哈希、启动参数和无后门策略与软件商签名完全一致——HSM 才会通过加密信道将 $K_{\text{model}}$ 直接释放进 TEE 飞地内存解密权重！一旦客户管理员试图开启 SSH 调试或修改容器代码，测量哈希（PCR/RIM）立即改变，HSM 拒绝下发密钥，模型永远无法被解密！

---

## ③ 🧭 直观可视化图解（Mermaid 决策树、TEE 信任边界图与远程证明 SKR 时序图）

### 3.1 企业级高机密大模型与模型 IP 保护架构选型决策树

```mermaid
flowchart TD
    A[企业计划部署高机密大模型或向客户交付私有核心模型 IP] --> B{核心合规与安全威胁模型？}
    B -->|常规企业内外部业务<br/>信任云厂商底层宿主机，仅需防公网与第三方留存| C[Private Link + 零数据留存 ZDR + CMK 磁盘加密<br/>详见 Round 366 企业混合云架构指南]
    B -->|主权政务 / 核心量化策略 / 芯片研发 / 医疗基因数据<br/>或 ISV 防止模型权重被运维 Root 窃取 (零信任威胁模型)| D{是否接受仅 5.8% 的硬件加密损耗？}
    D -->|是：追求 2 天极速上线 + 密码学可验证零信任<br/>且希望比自建机房省下 55% 预算| E[🟩 Tier 4: 零信任机密 AI 飞地 (Confidential AI + SKR)<br/>CPU SEV-SNP/TDX + GPU H100/H200 CC + 双端远程证明 + HSM 密钥释放<br/>100% 运行态内存与模型 IP 保护 / $27,800/月 🏆]
    D -->|否：法规死命令要求硬件物理放置在本单位大楼内| F[🟥 Tier 2: 本地物理隔离自建机房 + 机密计算服务器<br/>交付周期 9 个月 / 月均摊销 $62,000<br/>仍建议开启本地 GPU CC 防内部运维窃密]
```

### 3.2 Tier 4 零信任机密 AI 飞地（CPU TEE + GPU TEE + HSM SKR）信任边界架构图

```mermaid
flowchart LR
    subgraph Untrusted["🚫 不可信基础设施层 (云厂商超管 / 宿主机 OS / Kubernetes 节点 Root)"]
        U1["Host Hypervisor / VMM (看不见明文)"]
        U2["加密磁盘上的密文模型权重 (.safetensors.enc)"]
    end

    subgraph Enclave["🛡️ 硬件可信执行环境飞地 (Hardware Trusted Execution Environment)"]
        direction TB
        CPU["🔒 CPU TEE (AMD SEV-SNP / Intel TDX)<br/>硬件内存控制器 AES 加密 DRAM<br/>运行受签名锁死的 vLLM 机密容器 (禁 SSH/Exec)"]
        SPDM["⚡ PCIe SPDM 硬件加密隧道<br/>Bounce Buffer AES-GCM-256"]
        GPU["🔒 GPU TEE (NVIDIA H100 / H200 CC)<br/>关闭 JTAG/调试端口 · HBM3 显存硬件隔离<br/>80层 Transformer 100% 满血解密运算"]
        CPU <--> SPDM <--> GPU
    end

    subgraph TrustRoot["🔑 独立信任根与客户端验证"]
        ATT["📜 双端远程证明服务 (Verifier)<br/>核验 CPU + GPU 芯片签名与容器哈希"]
        HSM["🏛️ 客户专属硬件安全模块 (Managed HSM)<br/>仅向验证通过的飞地安全释放密钥 (SKR)"]
        CLIENT["👤 医疗/金融客户端<br/>验证飞地证明后建立端到端 mTLS 直连"]
    end

    CPU & GPU -->|1. 提交硬件证明 Quote| ATT
    ATT -->|2. 签发证明 JWT| HSM
    HSM -->|3. 安全密钥释放 (SKR) 直入飞地内存| CPU
    U2 -->|4. 读入飞地内实时解密至 GPU HBM| CPU
    CLIENT <-->|5. 端到端加密推理 (宿主机全程零可见)| CPU
```

### 3.3 双端远程证明（CPU+GPU Remote Attestation）与 HSM 密钥释放时序图

```mermaid
sequenceDiagram
    autonumber
    participant CVM as 机密容器 (CPU SEV-SNP + H100 CC)
    participant Verifier as 远程证明核验服务 (Attestation Service)
    participant HSM as 客户专属硬件安全模块 (HSM / KMS)
    participant User as 零信任企业客户端

    Note over CVM: 1. 容器启动，挂载磁盘上的密文模型权重 (此时无密钥无法解密)
    CVM->>CVM: 2. CPU 与 H100 GPU 完成芯片级 SPDM 双向握手与测量 (RIM/PCR)
    CVM->>Verifier: 3. 发送由芯片出厂私钥签名的 CPU Quote + GPU Attestation Report + 容器镜像哈希
    Verifier->>Verifier: 4. 校验芯片证书链、固件版本及容器策略 (确认已禁用 SSH/Debug 且镜像未篡改 ✅)
    Verifier-->>CVM: 5. 签发短效加密证明令牌 (Attestation Token)
    CVM->>HSM: 6. 携带证明令牌与飞地临时公钥请求安全密钥释放 (Secure Key Release, SKR)
    HSM-->>CVM: 7. 匹配密钥释放策略成功！将模型解密密钥 K_model 用飞地公钥封装下发
    CVM->>CVM: 8. 在硬件加密内存内解密 K_model，将 70B 权重直接注入 H100 CC 显存 🏆
    User->>CVM: 9. 客户端核验飞地证明证书后，建立直通飞地内部的 TLS 会话发送机密 Prompt
```

---

## ④ 📊 架构师多方案选型决策矩阵（Architect Decision Matrix）

| 评估维度 | 🟨 Tier 1：标准云端明文内存 GPU | 🟥 Tier 2：100% 本地自建物理机房 | 🟦 Tier 3：云端机密 VM + GPU CC | 🟩 Tier 4：零信任 TEE + 远程证明 + HSM SKR |
| :--- | :--- | :--- | :--- | :--- |
| **运行态内存与显存保护 (`Data-in-Use`)** | `0.0%` ✕（宿主机可 Dump 明文） | `95.0%`（物理隔离，但防不住内鬼） | `92.0%` ✅（CPU+GPU 硬件加密） | **`100.0%` 🏆（硬件飞地 + 密码学证明锁死）** |
| **私有模型权重防盗 (`Model-IP-in-Use`)** | ✕ 磁盘/内存明文易被拷走 | ✕ 本地 Root 运维可直接拷走 | ⚠️ 启动密钥未与容器哈希强绑定 | **🏆 密文镜像 + 远程证明通过后仅在飞地内解密** |
| **70B 大模型推理吞吐损耗** | `0.0%` | `0.0%` | `5.4%` ✅ | **`5.8%` 🏆（PCIe 加密被大模型计算完美掩盖）** |
| **从立项到上线交付周期** | `1 天` | **`270 天`（9 个月基建采购）✕** | `2 天` ✅ | **`2 天` 🏆（比自建机房快 `135x`）** |
| **突发流量弹性扩容能力** | ✅ 秒级/分钟级云弹性 | ✕ 零弹性（按峰值买卡长期闲置） | ✅ 云端按需扩容 | **🏆 云端按需扩容机密飞地节点** |
| **100 RPS 规模月均总成本 (TCO)** | `$24,000 / 月` | **`$62,000 / 月` ✕（重资产摊销）** | `$26,200 / 月` ✅ | **`$27,800 / 月` 🏆（比自建机房省 `55.2%`）** |

---

## ⑤ 🧪 容量/成本拐点算例与 7-Card 报告（Quantitative Sizer & 7-Card Deck）

### 5.1 基准算例假设（70B-FP8 高机密行业模型 · 峰值 100 RPS · 3 年期 TCO 对比）
- **为什么 Tier 4（`$27,800/月`）比 Tier 1 普通云主机（`$24,000/月`）只贵了 `15.8%`，却能比本地自建机房（`$62,000/月`）省下 `55.2%`？**
  1. **云端机密 GPU 溢价极低**：主流云厂商的 H100 机密计算实例（如 Azure `NCCads H100 v5`）相比普通 H100 实例溢价仅为 `5%~10%`，算上 `5.8%` 的 PCIe 加密吞吐损耗与托管 HSM 费用，总开销仅比裸奔云主机高出 **`15.8%`（每月多花 `$3,800`）**；
  2. **彻底省去自建机房的三大无底洞**：本地自建机房不仅要一次性垫付数百万美元买断硬件，还要为应对业务波峰多买 `40%` 冗余显卡（平时闲置吃灰），并长期雇佣专职硬件/网络/液冷值班团队。

```chart
{
  "data": {
    "values": [
      { "Tier": "1. 标准云端明文GPU", "运行态保护得分(%)": 0.0, "推理吞吐保持率(%)": 100.0, "月均总TCO($K)": 24.0 },
      { "Tier": "2. 本地自建物理机房", "运行态保护得分(%)": 95.0, "推理吞吐保持率(%)": 100.0, "月均总TCO($K)": 62.0 },
      { "Tier": "3. 云端CVM+GPU-CC", "运行态保护得分(%)": 92.0, "推理吞吐保持率(%)": 94.6, "月均总TCO($K)": 26.2 },
      { "Tier": "4. 零信任TEE+证明+SKR", "运行态保护得分(%)": 100.0, "推理吞吐保持率(%)": 94.2, "月均总TCO($K)": 27.8 }
    ]
  },
  "semantic_types": {
    "Tier": "Category",
    "运行态保护得分(%)": "Percentage",
    "推理吞吐保持率(%)": "Percentage",
    "月均总TCO($K)": "Amount"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "零信任机密 AI 飞地 (Tier 4) 实现 100% 运行态内存保护、94.2% 吞吐保持率且比自建机房省 55.2%",
    "subtitle": "70B高机密负载 (100 RPS) 下的运行态保护得分 (%)、推理吞吐保持率 (%) 与月均 TCO ($K)",
    "encodings": {
      "x": "Tier",
      "y": ["运行态保护得分(%)", "推理吞吐保持率(%)", "月均总TCO($K)"]
    }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（Production Readiness Checklist）

在企业上线机密计算大模型（Confidential AI）或向外部交付高价值私有模型前，架构师必须逐项核对以下 8 条铁律：

- [ ] **1. 必须同时启用“CPU 机密虚拟机（AMD SEV-SNP / Intel TDX）”与“GPU 硬件机密模式（NVIDIA H100/H200 CC）”**：严禁只开 CPU 加密却把数据明文传给未开启 CC 模式的普通 GPU。
- [ ] **2. 多卡张量并行（Multi-GPU TP）机密推理必须使用支持 `NVLink CC（机密 NVLink）` 的 8 卡机型**：利用 GPU 间硬件加密 NVLink 通道传输跨卡 AllReduce 激活值，避免回退走慢速 PCIe。
- [ ] **3. 容器运行态策略必须强制关闭交互式后门（`allow_exec: false` & 禁用 SSH）**：在机密容器（Kata Containers CoCo / Azure ACI Confidential）策略清单中硬性禁止任何宿主机管理员 `kubectl exec` 进入容器读取显存。
- [ ] **4. 私有模型权重文件在对象存储与容器镜像中必须全程 `AES-256` 加密存放**：严禁将明文 `.safetensors` 打包进 Docker 镜像或挂载在未加密云盘上。
- [ ] **5. 强制实施“先双端远程证明（CPU+GPU Attestation），后释放密钥（HSM SKR）”**：HSM 密钥释放策略必须同时校验 CPU 测量值（PCR）与 NVIDIA GPU 固件完整性报告（RIM）。
- [ ] **6. TLS 终结（TLS Termination）必须下沉至 TEE 飞地容器内部（Attested TLS / RA-TLS）**：严禁在飞地外部的普通 Nginx / 负载均衡器上提前解密 HTTPS 流量（那会让明文暴露在飞地外！），外部网关只能做四层 TCP/SNI 透传。
- [ ] **7. 禁用或重定向飞地内部的标准输出日志（`stdout/stderr` 零敏感日志）**：确保推理引擎不会把用户的明文 Prompt 或输出 Token 打印到宿主机可见的容器日志采集器中。
- [ ] **8. 开启批处理与请求聚合以摊薄 PCIe Bounce Buffer 加解密开销**：保持生产 Continuous Batching，使 GPU 机密计算端到端损耗稳定控制在 `< 6%` 以内。
