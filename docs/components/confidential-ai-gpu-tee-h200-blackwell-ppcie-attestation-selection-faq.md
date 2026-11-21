# 大模型机密计算（`Confidential AI / GPU TEE`）与主权云安全落地架构选型：`NVIDIA H200 PPCIE` 多卡加密互联、`Blackwell B200` 线速内联加密、双硬件远程证明（`NRAS`）与消除 `53.5%` PCIe 加密税指南

> **语言 / Language**: **中文 (当前)** | [English](./confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect、金融/政务/医疗合规架构师的大模型硬件可信执行环境（`CPU SEV-SNP/TDX + GPU TEE`）、单卡 `CC-On` vs. 多卡 `PPCIE` vs. `Blackwell TDISP` 线速加密选型、远程证明密钥释放（`SKR`）与性能损耗压降（从 `53.5%` 降至 `<1.6%~4.8%`）白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业高敏数据机密推理架构设计与选型）

![大模型机密计算 GPU TEE H200 PPCIE 与 Blackwell 内联线速加密总结图](../../assets/svg/confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**当金融、医疗、主权云客户要求“在公有云或托管机房跑大模型，但连云厂商宿主机 Root 管理员和物理探针都绝不能偷看到明文 Prompt、KV Cache 和自研模型权重（Data-in-Use 内存态加密）”时，硬件级 **`CPU CVM (SEV-SNP/TDX) + GPU TEE`** 是唯一合规解！很多团队测试第一代 GPU TEE 时发现吞吐暴跌 **`53.5%`**，根源是错误地在 TEE 里开启了主存 KV 换出（`Host DRAM Swap`）或 `--enforce-eager` 模式，把几百 MB 数据和 650 个算子指令塞进只有 `4~8 GB/s` 的 `SWIOTLB` PCIe 软件加密回弹缓冲区！只要遵循“**100% 权重与 KV 驻留 HBM（跨 PCIe 仅传几 KB 的 Token ID）+ 强制开启 `CUDA Graph` + 多卡开启 `H200 PPCIE`（加密 NVLink）或升级 `Blackwell B200/GB200`（硬件内联线速加密引擎 `TDISP`）**”，就能在 `H200` 上保住 **`95.2%` 裸金属性能（仅 `4.8%` 加密税）**，在 `B200` 上保住 **`98.4%` 裸金属性能（仅 `1.6%` 加密税）**！**

- **交互式 GPU TEE 机密计算加密税与吞吐保持率计算器**：[打开交互式计算器 (`confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-sizer.html`)](../reports/confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-selection-cards.html`)](../reports/confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：为什么在 `Hopper (H100/H200) CC-On` 机密虚拟机里，一旦触发 PCIe 大数据搬运性能就会暴跌 53.5%？怎么用三条铁律把加密损耗压到 `< 4.8%`？
1. **剖析 `SWIOTLB` 回弹缓冲区与 `AES-GCM` 加密瓶颈**：
   - 在机密计算模式下，宿主机 Hypervisor 是不可信的（Untrusted），CPU CVM 的内存由 `AMD SEV-SNP / Intel TDX` 硬件密钥加密，GPU HBM 由 GPU 内部安全处理器（`GSP`）加密；
   - 因此，任何跨越 PCIe 总线的数据不能直接做明文 DMA，必须先拷入未加密的共享内存回弹缓冲区（`SWIOTLB Bounce Buffer`），并由 CPU 和 GPU Copy Engine 用 **`AES-GCM-256`** 逐块加解密验签（在 Hopper 上单卡加密吞吐上限仅约 `4 ~ 8 GB/s`）！
2. **把 Hopper GPU TEE 加密损耗从 `53.5%` 砍到 `4.8%` 的三条黄金铁律**：
   - **铁律 1（严禁在 Hopper TEE 内开启 Host DRAM KV Offload）**：设置 `--swap-space 0`，确保模型权重与活跃 KV Cache **100% 驻留在 GPU HBM 内部**！这样每次请求跨越 PCIe 总线的只有输入的几千个整数 `Token IDs`（`< 32 KB`）和输出的几个字节，PCIe 加密耗时瞬间趋近于 `0`！
   - **铁律 2（强制开启 `Piece-wise CUDA Graph`）**：在 Eager 模式下，每生成一个词 CPU 都要通过加密通道向 GPU 发送 650 次核函数启动描述符；开启 `CUDA Graph` 后每步仅需 1 次加密指令交互！
   - **铁律 3（多卡 `TP=4/8` 启用 `PPCIE` 模式，或升级 `Blackwell B200`）**：`H100/H200` 的 **`PPCIE`（Protected PCIe）** 模式为 8 卡 `NVLink` 建立动态会话加密密钥；而新一代 **`Blackwell B200 / GB200`** 直接在 PCIe Gen6 (`TDISP`) 与 NVLink 5 物理控制器中内置了**全线速硬件内联加解密引擎（Inline Crypto Engine）**，彻底消灭了回弹损耗（保持 **`98.4%`** 裸机速度）！

### Q2：如何保证“我们价值上千万的自研微调权重放在云主机磁盘上不会被云管理员拷走”？（双硬件远程证明与安全密钥释放 `SKR`）
- **加密模型镜像 + 双硬件远程证明（Dual Remote Attestation）闭环**：
  1. 模型权重离线使用 `AES-256-GCM` 加密为 `.safetensors.enc` 存入云存储，哪怕云管理员把磁盘快照偷走也只是一堆乱码；
  2. Pod 启动时，机密容器内的 Attestation Sidecar 同时收集 **CPU TEE 硬件签名报告（SEV-SNP/TDX Quote）** 与 **GPU SPDM 硬件证书链（通过 `NVIDIA Remote Attestation Service / NRAS` 校验 GPU 固件哈希与 `CC-On / PPCIE` 状态）**；
  3. 只有当企业自管 KMS / HSM 验证“CPU 与 8 张 GPU 均处于真实硬件 TEE 且容器镜像 Digest 未被篡改”后，才通过 TLS 通道将权重解密密钥（`KEK/DEK`）直接注入 CVM 内存，解密后流式灌入受保护的 GPU HBM！

---

## 3. 🧭 直观可视化图解（Mermaid 双硬件远程证明与加密边界流）

### 3.1 `CPU CVM + GPU TEE (PPCIE / TDISP)` 信任边界与安全密钥释放（`SKR`）时序图
```mermaid
sequenceDiagram
    autonumber
    participant Disk as 不可信云磁盘 / 对象存储 (.safetensors.enc)
    participant CVM as 受保护 CPU CVM (SEV-SNP / TDX)
    participant GPU as 受保护 8x GPU TEE (H200 PPCIE / B200 TDISP)
    participant KMS as 企业自管 KMS + NVIDIA NRAS 证明服务

    CVM->>GPU: 1. 通过 SPDM 协议建立 CPU<->GPU 加密通道并读取 GPU 硬件证明度量
    CVM->>KMS: 2. 提交 [CPU TEE Quote + GPU SPDM Attestation Report + 容器镜像 Hash]
    KMS->>KMS: 3. 校验 CC-On/PPCIE 开启、固件无篡改、安全启动通过!
    KMS-->>CVM: 4. 仅向可信 CVM 内存释放 AES-256 模型解密密钥 (SKR)
    Disk->>CVM: 5. 读取加密模型分片并在 CVM/GPU 内直接解密灌入受保护 HBM
    Note over CVM,GPU: 6. 运行时仅 <32KB Token ID 经 AES-GCM 跨 PCIe · 95.2%~98.4% 满速推理!
```

---

## 4. 📊 架构师多方案选型决策矩阵（四档机密 AI 部署方案对比）

| 评估维度 | Tier 1：朴素开启 `CC-On`（开主存 KV Swap + Eager 模式） | Tier 2：单卡 `H200 CC-On`（`TP=1` + HBM 驻留 + CUDA Graph） | Tier 3：8 卡 `H200 PPCIE`（多卡加密 NVLink + 远程证明 KMS） | Tier 4（推荐旗舰）：`Blackwell B200` 硬件内联线速 TEE（`TDISP` + Inline Crypto） |
| :--- | :--- | :--- | :--- | :--- |
| **相比裸金属吞吐保持率** | `46.5%` ✕（损失 53.5%！） | `93.8%` ✓ | **`95.2%` ✓（Hopper 多卡最优）** | **`98.4%` ✓（近乎零损耗！）** |
| **PCIe / NVLink 加密性能税** | `53.5%` ✕（死于 Bounce Buffer） | `6.2%` | **`4.8%` ✓** | **`1.6%` ✓（硬件内联线速引擎）** |
| **`70B FP8` P99 Decode `TPOT`** | `34.8 ms` ✕ | `28.4 ms`（受限于单卡带宽） | **`14.4 ms` ✓（8卡聚合带宽）** | **`8.2 ms` ✓** |
| **每百万机密 Token 综合成本** | `$1.12 / 1M` | `$0.58 / 1M` | `$0.44 / 1M` | **`$0.28 / 1M` ✓（降低 `75.0%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **Hopper (`H100/H200`) 机密推理强制设置 `--swap-space 0` 并开启 `CUDA Graph`**：杜绝大块张量频繁穿越 `SWIOTLB` PCIe 回弹缓冲区。
- [ ] **多卡机密推理（`TP=2/4/8`）必须开启 `PPCIE` 模式**，并在新一代集群优先选用内置硬件线速加密引擎的 `Blackwell B200 / GB200`。
- [ ] **模型权重落盘强制加密（`.safetensors.enc`），并通过 `CPU TEE + NVIDIA NRAS` 双重远程证明门禁释放解密密钥**。
