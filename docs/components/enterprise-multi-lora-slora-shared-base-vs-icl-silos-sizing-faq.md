# 企业百部门微调模型共池服务（`Multi-LoRA Serving · 128 个垂域 Adapter 共享 1 份 70B 基座`）显存手算、统一分页适配器池（`S-LoRA / Punica / vLLM V1`）与 `16K Few-Shot ICL` / 独立合并部署对比指南（`#636`）

> **Language / 语言**: **简体中文** | [English](./enterprise-multi-lora-slora-shared-base-vs-icl-silos-sizing-faq.en.md) · [📱 交互式 H5 精读页（含 LoRA 显存手算器 + 百部门共池沙盘）](../h5/multi-lora-slora-shared-base-vs-icl-silos-lab.html) · [🧮 交互式容量计算器 Sizer](../reports/enterprise-multi-lora-slora-shared-base-vs-icl-silos-sizing-sizer.html) · [🃏 7-Card 评测报告](../reports/enterprise-multi-lora-slora-shared-base-vs-icl-silos-sizing-cards.html) · [返回总目录](../full-catalog-zh.md)

![架构摘要图](../../assets/svg/enterprise-multi-lora-slora-shared-base-vs-icl-silos-sizing-faq-summary.svg)

---

## 一、3 分钟核心结论卡片（Executive Summary）

1. **一句话结论**：
   - 当大型企业的法务、财务、HR、客服、20 个产品线各自微调了 **`128 个` 垂域专属模型** 时，若采用传统的 **“把 LoRA 权重合并回 70B 底座、一部门部署一套独立实例（Merged Weight Silos）”**，哪怕每个部门只配最小 `2 张 H200`，也需要 **`256 张 H200`（月烧 `$819,200` / 年烧 `$983万`）**，且 90% 的部门专卡全天处于空转状态！而若退而求其次不微调，改用 **每次请求硬塞 `16K Few-Shot 示例（ICL）`**，不仅复杂垂域格式遵从率跌至 `82.4%`，还会让单请求 KV Cache 暴涨 `20 倍`（仍需 **`24 张 H200`**）。
   - 采用 **`vLLM V1 / SGLang (S-LoRA / Punica)` 动态多适配器共池架构（Multi-LoRA Serving）**：将 **`1` 份只读的 `70B FP8` 基础权重常驻在 `8 张 H200`（2 组 `TP=4`）显存中**，所有 128 个部门的请求在同一个 Batch 内**共享执行主干大矩阵乘法**，仅在旁路通过 `SGMV` 算子叠加各自的 `266 MB` 轻量 LoRA 增量（$\Delta y = x A_i B_i$），将总卡数从 `256 张` 压降至 **`8 张 H200`（立省 `96.9%` GPU，每年净省 `$952.3万`）**，同时保持 **`97.6%` 垂域准确率** 与 **`14.8 ms` 的丝滑 `P99 TPOT`**！

2. **单个 LoRA Adapter 显存体积精确手算公式（架构师白板必背）**：
   - 对层数 $L=80$ 的 `Llama-3.3-70B`（隐藏维 $d_{\text{model}}=8192$，FFN 中间维 $d_{\text{ffn}}=28672$，GQA KV 头维 $8 \times 128 = 1024$），当对全部 `7` 个线性投影层（`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`）注入秩为 **$r = 32$**、精度为 `BF16`（$2\text{ Bytes}$）的 LoRA 矩阵 $(A, B)$ 时：
     $$M_{\text{LoRA}}(r) = L \times 2\text{ Bytes} \times r \times \left[ \underbrace{4(d + d)}_{Q,O} + \underbrace{2(d + d_{kv})}_{K,V} + \underbrace{3(d + d_{\text{ffn}})}_{\text{Gate,Up,Down}} \right]$$
     - 代入数值：$80 \times 2 \times 32 \times [65,536 + 18,432 + 110,592] / 2 \approx \mathbf{266.2\text{ MB / 个 Adapter}}$（若 `r=16` 则仅 **`133.1 MB`**）！
     - **惊人对比**：`1` 份完整 `70B BF16` 权重占 `140,000 MB`，而 **`1` 个 `r=32` 的 LoRA 仅占 `266 MB`（相差 `526 倍`）**！在单机 HBM 中预留区区 **`8.5 GB` 适配器槽位**，就能同时常驻 **`32 个` 最活跃部门的微调模型**，其余 96 个长尾适配器常驻在主机 Pinned DDR5 中，按需经 `PCIe Gen5` 换入仅需 **`1.8 ms`**！

```mermaid
flowchart LR
    subgraph Gateway["🧭 Envoy GIE 网关 (lora_id 亲和路由)"]
        REQ["128 个部门并发请求\n(法务合同 / 财务税表 / 醫療病历 / 客服)"] --> ROUTE["按 model='dept-lora-i' 哈希亲和\n优先路由至已驻留该 LoRA 的 TP=4 实例"]
    end

    subgraph Engine["⚡ vLLM V1 / SGLang (S-LoRA Unified Paging · 8x H200)"]
        BASE["共享只读 70B FP8 基座权重 (70 GB)\n64 路异构部门请求拼成 1 个大 Batch\n统一跑 Tensor Core 主干 GEMM y_0 = x·W_0"]
        LORA["Punica SGMV / BGMV 旁路内核\n并行计算各请求专属增量 Δy_i = x_i · A_i · B_i\nHBM 驻留 Top-32 热 LoRA (仅占 8.5 GB)"]
        SWAP["Host Pinned DDR5 存放全部 128 个 LoRA (34 GB)\n冷门 Adapter 按需异步换入仅需 1.8 ms!"]
    end

    ROUTE --> BASE & LORA
    SWAP -.->|"PCIe Gen5 零拷贝换入"| LORA
```

---

## 二、三种多部门定制化落地方案定量对比矩阵（`128 个垂域任务 · 峰值 30 QPS`）

| 评估维度 | 方案 A：一部门一合并实例 (`128 × 2 = 256x H200`) | 方案 B：共享基座 + 硬塞 `16K Few-Shot ICL` (`24x H200`) | 方案 C：`S-LoRA / vLLM` 共享 70B 基座 + 动态分页适配器池 (`#636`) 🏆 |
| :--- | :--- | :--- | :--- |
| **垂域复杂格式与知识准确率** | `97.6%` | `82.4%`（示例冲突与注意力分散 ✕） | **`97.6%`（与独立合并微调模型数学等价 ✓）** |
| **单请求平均输入 Token 长度** | `800 Tokens` | `16,800 Tokens`（硬塞 `16K` 示例 ✕） | **`800 Tokens`（领域知识已内化进 `266MB` LoRA ✓）** |
| **首字延迟 (`P95 TTFT`) 与 `P99 TPOT`** | `210 ms` / `13.4 ms` | `1,380 ms` / `22.6 ms` | **`225 ms` / `14.8 ms`（SGMV 旁路仅 `+1.4ms` ✓）** |
| **所需 H200 总卡数与月度账单** | `256× H200`（`$819,200/月` ✕） | `24× H200`（`$76,800/月`） | **`8× H200`（`$25,600/月`，较独立部署立省 `96.9%` ✓）** |

---

## 三、Multi-LoRA 生产上线核对清单（SA Pre-Production Checklist）

- [ ] **1. 统一各部门微调基座版本与最大秩（`max_lora_rank = 32 / 64`）**：全公司制定统一的基座模型标准（如 `Qwen2.5-72B-Instruct-FP8`），禁止各部门自行修改基座词表或采用超过 `r=64` 的冗余大秩；
- [ ] **2. 在 `vLLM / SGLang` 启动参数中显式声明热缓存槽位**：设置 `--enable-lora --max-loras 32 --max-cpu-loras 128 --max-lora-rank 32`，并结合 `Envoy GIE` 按 `lora_id` 做一致性哈希路由。
