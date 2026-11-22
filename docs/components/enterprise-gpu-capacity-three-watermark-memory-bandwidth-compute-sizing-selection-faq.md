# 企业大模型 GPU 容量规划（Sizer）三道水位线终极手算指南：第一道显存水位（权重 + `MHA/GQA/MLA` KV Cache + `85%` 规划线）、第二道 HBM 带宽水位（最低 `TPOT`）与第三道算力并发水位（`L40S / H100 / H200 / B200 / MI325X` 对标）

> **语言 / Language**: **简体中文** | [English](./enterprise-gpu-capacity-three-watermark-memory-bandwidth-compute-sizing-selection-faq.en.md)  
> **配套交互式学习与精算工具**: [📱 交互式 H5 精读页 (三道水位线 GPU 容量手算与显存堆叠瀑布沙盘)](../h5/gpu-three-watermark-memory-bandwidth-compute-sizer-lab.html) · [🧮 HTML 交互式计算器](../reports/enterprise-gpu-capacity-three-watermark-memory-bandwidth-compute-sizing-selection-sizer.html) · [🃏 7-Card 4:5 视觉评测卡](../reports/enterprise-gpu-capacity-three-watermark-memory-bandwidth-compute-sizing-selection-cards.html) · [📊 可复核模型](../../experiments/enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection/) · [🎓 返回 AI Infra 全景精学馆](../h5/index.html)

![企业大模型 GPU 容量规划三道水位线手算全景图](../../assets/svg/enterprise-gpu-capacity-three-watermark-memory-bandwidth-compute-sizing-selection-faq-summary.svg)

---

## 一、3 分钟看懂核心结论（Executive Summary）

> **一句话决策结论**：回答客户“部署这个大模型到底需要买多少张 GPU？”时，**最致命的错误是“只拿模型权重大小除以单卡显存（如 `70B BF16 = 140GB` $\implies$ 买 2 张 `H100 80GB`，`Tier 1`）”**！这种拍脑袋算法上线后，扣掉 `15%` 架构安全余量与 CUDA Graph 激活值后只剩 `10GB` 装 KV Cache，进 4 个 `16K` 并发就会触发排队驱逐，`P99 TPOT` 飙至 **`48.2 ms`**！同样，只看显存容量选用 `4x L40S 48GB GDDR6`（`Tier 2`），虽能装下权重，但因 **GDDR6 带宽仅 `864 GB/s` 且无 NVLink**，流式吐字仍卡在 `34.6 ms`！
> - **架构师三道水位线取最大值总公式（`Tier 4`）**：
>   $$N_{\text{GPU}} = \max\!\Big(\underbrace{\left\lceil \frac{M_{\text{weight}} + M_{\text{KV}} + M_{\text{act+graph}}}{0.85 \times M_{\text{HBM}}} \right\rceil}_{\text{① 第一道水位线：装得下 (显存容量)}},\; \underbrace{\left\lceil \frac{M_{\text{active\_weight}} + M_{\text{KV\_read}}}{\text{TPOT}_{\text{SLA}} \times \text{BW}_{\text{eff}}} \right\rceil}_{\text{② 第二道水位线：吐得快 (HBM 带宽)}},\; \underbrace{\left\lceil \frac{\text{RPS}_{\text{peak}} \cdot S_{\text{in}}(1-\eta) \cdot 2 P_{\text{active}}}{\text{TFLOPS}_{\text{eff}} \times 10^{12}} \right\rceil}_{\text{③ 第三道水位线：读得完 (Prefill 算力)}}\Big)$$
> - **满血 `DeepSeek-V3/R1 671B (含 14B MTP 头 = 685GB FP8)` 硬件定容黄金结论**：
>   - 用 `H100 80GB`（`Tier 3`）：单机 8 卡仅 `640 GB`，连 `685 GB` 权重都装不下，被迫上 **双机 16 卡（`1,280 GB`）** 并承担跨机 RoCE/IB 通信税（`P99 TPOT = 15.8 ms`，`$0.29/1M`）；
>   - 用 **`单机 8x H200 141GB (1,128 GB HBM3e)` 或 `8x DGX B200 180GB (1,440 GB)`（`Tier 4`）**：单台 8 卡机器一步到位装下 **`685 GB` 权重 + `108 GB`（`192 路 × 16K` MLA KV Cache）+ `36 GB` 激活/图 + `15%` 规划余量**，100% 锁死在机内 NVLink 域，`P99 TPOT` 压至 **`11.4 ms`**，单百万达标 Token 成本降至 **`$0.16`（比 Tier 1 省 `86.4%`）**！

| 核心选型维度 | `Tier 1: 仅按权重定容 (2x H100 跑 70B BF16)` | `Tier 2: 4x L40S 48GB GDDR6 (无 NVLink)` | `Tier 3: 双机 16x H100 80GB (跑 671B+14B MTP)` | `Tier 4: 三道水位线配平单机 8x H200 141GB / B200 (🏆)` |
|---|---|---|---|---|
| **配置 GPU 卡数与总显存** | `2 张 (160 GB HBM3)` | `4 张 (192 GB GDDR6)` | `16 张 (1,280 GB HBM3 · 跨 2 机)` | **`8 张 (1,128 GB HBM3e · 单机 NVLink)`** 🏆 |
| **可承载 `16K` 并发活跃会话数** | **`4 路` ✕ (KV 显存饥饿)** | `28 路` | `160 路` ✓ | **`192 路` (含 `15%` 安全水位预留)** 🏆 |
| **流式吐字延迟 (`P99 TPOT`)** | `48.2 ms` ✕ | **`34.6 ms` ✕ (864GB/s 带宽受限)** | `15.8 ms` ✓ | **`11.4 ms` (`4.8 TB/s/卡` HBM3e)** 🏆 |
| **SLA 达标率 (`TTFT & TPOT`)** | `18.5%` ✕ | `46.0%` ✕ | `99.1%` ✓ | **`99.8%`** 🏆 |
| **每百万达标 Token 成本 (`$/1M`)** | `$1.18 / 1M` ✕ | `$0.64 / 1M` | `$0.29 / 1M` ✓ | **`$0.16 / 1M` (-86.4%)** 🏆 |

---

## 二、核心技术原理与架构师灵魂拷问（Q&A）

### Q1：第一道水位线（显存容量 $N_{\text{mem}}$）的四笔账到底怎么一分不差地手算？为什么生产规划必须除以 `0.80 ~ 0.85`？
1. **单实例总显存需求的四件套拆解**：
   - **① 模型静态权重（$M_{\text{weight}}$）**：$\text{参数量 } P \times \text{每参数字节数}$。
     - `Llama-3.3-70B`：`BF16` 为 `140 GB`；`FP8` 为 **`70 GB`**；`W4A16 AWQ` 为 **`38 GB`**。
     - `DeepSeek-V3/R1 671B`：**严禁漏算原生 `14B MTP` 投机预测头**！主模型 `671 GB` + MTP 头 `14 GB` = **`685 GB`（FP8）**。
   - **② 并发活跃 KV Cache（$M_{\text{KV}} = B_{\text{active}} \times S_{\text{ctx}} \times \text{KV}_{\text{per\_tok}}$）**：
     - **传统 70B MHA（FP8）**：`1,310.7 KB/tok`（16K 单会话占 `21.47 GB`）；
     - **主流 70B GQA-8（FP8）**：`163.84 KB/tok`（16K 单会话占 `2.68 GB`，64 并发占 **`171.8 GB`**）；
     - **DeepSeek 671B MLA（FP8）**：每层仅缓存 `512` 维压缩潜向量 $c_t^{KV}$ 与 `64` 维解耦 RoPE $k_t^R$，61 层每 Token 仅 **`35.14 KB`**（16K 单会话仅占 **`0.56 GB`**，192 并发仅占 **`108.0 GB`**！）。
   - **③ 运行时激活缓冲区、Piece-wise CUDA Graph 与 NCCL 通信 Buffer（$M_{\text{act+graph}}$）**：通常占 **`16 ~ 36 GB`**（每卡 `2 ~ 4.5 GB`）。
   - **④ 为什么架构规划必须除以 `0.80 ~ 0.85`（预留 `15%~20%` 安全水位）**：虽然推理引擎（如 vLLM）默认参数可能设为 `gpu_memory_utilization=0.90`，但在生产架构规划中，为了吸收长尾请求的瞬时输出暴涨、碎片开销与 PyTorch 内存分配器峰值，**架构师容量规划水位必须按 `80%~85%` 折算**！

### Q2：第二道水位线（HBM 带宽 $N_{\text{bw}}$）怎么手算？为什么 MoE 模型在 `Batch=64` 时不能按 `37B` 激活参数算带宽？
1. **Decode 阶段 HBM 带宽下界公式**：
   每生成一个 Token，GPU 必须把所有被当前 Batch 触碰到的权重与全部在途会话的 KV Cache 从 HBM 完整读入寄存器一次：
   $$\text{TPOT}_{\min} = \frac{M_{\text{active\_weights}}(B) + M_{\text{KV}}(B)}{N_{\text{GPU}} \times \text{BW}_{\text{peak}} \times \eta_{\text{bw}}} + T_{\text{comm\_exposed}}$$
2. **MoE 专家激活并集效应（Coupon-Collector Union Effect）纠偏**：
   - 很多初学者误以为 `DeepSeek 671B` 每 Token 只激活 `37B` 参数，所以在 `Batch=64` 时也只读 `37 GB` 权重——**大错特错！**
   - 当并发 $B=1$ 时确实只读 `37 GB`；但当并发 $B=64$ 个不同请求同时路由到每层 256 个专家中的 Top-8 时，64 个请求选中的专家并集几乎会覆盖 `40%~65%` 的路由专家（等效需从 HBM 读取 **`120 ~ 180 GB`** 权重）！
   - 在 `8x H200`（单卡峰值 `4.8 TB/s`，实测有效带宽利用率 $\eta_{\text{bw}} \approx 72\%$，8 卡有效总带宽 `27.65 TB/s`）上，读取 `150 GB` 权重 + `108 GB` KV 耗时约 `9.3 ms`，加上 `1.1~2.1 ms` 通信与调度开销，实测 `P99 TPOT` 稳定在 **`11.4 ms`**！

```mermaid
flowchart LR
    Input["📥 业务输入：模型 (70B/671B)<br/>并发 B_active、上下文 S_ctx、SLA (TTFT/TPOT)"] --> W1["① 第一道：显存容量水位 N_mem<br/>(权重 + B*S*KV_tok + 激活图) / (0.85 * HBM)"]
    Input --> W2["② 第二道：HBM 带宽水位 N_bw<br/>(并发激活权重并集 + KV) / (TPOT_SLA * BW_eff)"]
    Input --> W3["③ 第三道：Prefill 算力水位 N_cmp<br/>Peak_RPS * S_in * (1-eta) * 2P / TFLOPS_eff"]
    W1 & W2 & W3 --> Max["🏆 最终实例/集群 GPU 卡数<br/>N_GPU = max(N_mem, N_bw, N_cmp) (向 2/4/8 幂次对齐)"]
```

---

## 三、主流企业级推理 GPU 关键规格对照表（严谨纠偏版）

| GPU 型号 | 显存容量与介质 | 单卡显存带宽 | 机内互联 (`Scale-Up`) | 架构师核心认知与选型定位 |
|---|---|---|---|---|
| **NVIDIA L40S** | `48 GB GDDR6` | `864 GB/s` | **无 NVLink（仅 PCIe Gen4 64GB/s）** | 适合 `7B~14B` 小模型、ViT 视觉编码或图像生成；跑 70B+ Decode 严重受限于 GDDR6 带宽 |
| **NVIDIA H100 SXM** | `80 GB HBM3` | `3,350 GB/s` | `NVLink 4 (900 GB/s 双向)` | `70B FP8` 主力卡，以及 PD 分离中的高算力 **Prefill 专用池** |
| **NVIDIA H200 SXM** | `141 GB HBM3e` | `4,800 GB/s` | `NVLink 4 (900 GB/s 双向)` | **单机 8 卡 (`1,128 GB`) 即可跑满血 `671B+14B MTP FP8`**，当前企业性价比王牌 🏆 |
| **NVIDIA B200 (DGX B200)** | **`180 GB / 卡` 交付** (芯片设计上限 `192 GB`) | `8,000 GB/s` | `NVLink 5 (1,800 GB/s 双向)` | 原生支持 `NVFP4`，单机显存与带宽双冠王（注意区分芯片 192GB 与 DGX 180GB 规格表） 🏆 |
| **AMD Instinct MI325X** | `256 GB HBM3e` | `6,000 GB/s` | `Infinity Fabric (896 GB/s)` | 单机 8 卡高达 `2,048 GB` 显存，超长上下文巨型 KV 池与满血 MoE 利器 |

---

## 四、生产落地 Checklist（Pre-Production Checklist）

- [ ] **禁止漏算附加模块显存**：部署 `DeepSeek-V3/R1 671B` 开启原生投机解码时，必须在 `671 GB` 主权重之外额外计入 **`14 GB` 的 MTP 模块权重**；部署多模态 VLM 时额外计入 `ViT` 视觉编码器与图像特征缓存。
- [ ] **区分 `L40S` 与 `H100/H200` 物理边界**：切勿将无 NVLink、仅 `864 GB/s` GDDR6 的 `L40S` 用于高并发 `70B+` 实时流式 Decode。
- [ ] **严格按三道水位线取最大值**：短文本高并发看 **第三道（算力）**，低并发极速流式看 **第二道（HBM 带宽）**，长上下文/多轮思维链看 **第一道（KV 显存）**。

---

## 五、来源与方法论溯源（Provenance）

- **数据来源声明**：本指南实验数据来自可复核解析模型 [`experiments/enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection/run.py`](../../experiments/enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection/run.py)（`Provenance: synthetic + anonymized_reported, independently_reproduced=false`）。
