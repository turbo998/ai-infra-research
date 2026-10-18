# 下一代超节点架构演进：从 Hopper (`NVL8`)、Blackwell (`GB200 NVL72`) 到 Vera Rubin (`VR200 NVL144` / `HBM4` / `NVLink-6` / `1.6T CPO`) 的 Roofline 与万亿 MoE 经济学

> **Evidence Status**:
> - **Specification & Architecture Roadmap**:
>   - JEDEC `JESD270-4` (HBM4 Standard Specification: 双倍通道位宽 **2048-bit per stack** vs HBM3/HBM3e 1024-bit per stack) 与 NVIDIA GTC 硬件路线图（Hopper `H100 SXM5 NVL8` $\rightarrow$ Blackwell `GB200/GB300 NVL72` $\rightarrow$ Vera Rubin `VR200 NVL144` / Rubin Ultra `NVL576`）；
>   - 结合本仓库已核验的 `experiments/ai_factory_tco_simulator/`、`experiments/cpo_scaleup_power_tco/` 与 `experiments/blackwell_fp4_mla_kernel/` 建立三代超节点统一算术强度拐点（Ridge Point）与 1T MoE 端到端 Token 经济学模型。
> - **Own-Offline Simulation**: `experiments/vera_rubin_hbm4_nvl144_roofline/run.py`（确定性离线仿真：对比 `Hopper H100 NVL8`、`Blackwell GB200 NVL72` 与 `Vera Rubin VR200 NVL144` 在 `1T MoE (64B active, 256 experts top-8, BS=128, L_kv=32,768 MLA)` 与 `1M` 超长上下文下的算术强度拐点、EP All-to-All 通信延迟、Decode TPOT 及每百万 Token 成本 `$/1M tokens`）。

---

## 1. 三代 AI 超节点硬件基本面演进：为什么 `HBM4 (2048-bit)` 与 `NVL144` 是分水岭？

在大模型从千亿稠密（Dense 70B–405B）全面转向**万亿参数细粒度稀疏混合专家（1T–2T MoE，如 DeepSeek-V3/R1 演进架构与 Kimi-K2，`256–384` 个路由专家）**以及 **`1M–4M` 超长上下文 Agentic 推理**的过程中，硬件瓶颈经历了三次范式转移：

| 架构维度 | **Hopper `H100 SXM5 NVL8`** | **Blackwell `GB200 NVL72`** | **Vera Rubin `VR200 NVL144`** | 架构跃迁核心物理意义 |
| :--- | :--- | :--- | :--- | :--- |
| **HBM 世代与单栈位宽** | HBM3 (`1024-bit/stack`, 5120-bit 总宽) | HBM3e (`1024-bit/stack`, 8192-bit 总宽) | **HBM4 (`2048-bit/stack`, 双倍引脚密度)** | HBM4 将单堆栈接口位宽翻倍至 `2048-bit`，突破引脚频率功耗墙 |
| **单 GPU 显存容量 / 带宽** | `80 GB` @ **`3.35 TB/s`** | `192 GB` @ **`8.00 TB/s`** | **`288 GB` @ `13.50 TB/s`** | Decode 访存带宽较 H100 提升 **`4.03x`**，较 GB200 提升 **`1.69x`** |
| **单 GPU 稠密算力 (FP8 / FP4)** | `1,979 TFLOPS` (FP8) / 无原生 FP4 | `4,500` (FP8) / **`9,000 TFLOPS` (FP4)** | `8,000` (FP8) / **`16,000 TFLOPS` (FP4)** | 原生 `NVFP4` 微缩放张量核使 Prefill 吞吐较 H100 跃升 **`8.08x`** |
| **算术强度拐点 $\mathcal{I}_{\text{ridge}}$** | `590.7 FLOP/B` (FP8) | `562.5` (FP8) / `1,125.0 FLOP/B` (FP4) | `592.6` (FP8) / **`1,185.2 FLOP/B` (FP4)** | FP4 下算术强度拐点翻倍至 `~1,185 FLOP/B`，倒逼 MLA 矩阵吸收与高密 MoE 拼批 |
| **Scale-Up 域规模与双向带宽** | **`NVL8`** (`900 GB/s` NVLink-4) | **`NVL72`** (`1,800 GB/s` NVLink-5) | **`NVL144`** (**`3,600 GB/s`** NVLink-6) | 单机柜 Scale-Up 域算力与总 HBM 容量（`41.4 TB`）足以容纳多副本万亿 MoE |
| **Scale-Out 网卡与光互连** | `400 Gbps` InfiniBand / RoCEv2 | `800 Gbps` ConnectX-8 | **`1.6 Tbps` ConnectX-9 + CPO 共封装光学** | 消除可插拔 DSP 重定时器功耗墙，支撑十万卡级低抖动 Scale-Out |

---

## 2. 万亿参数 MoE (`1T Total / 64B Active`) 的跨机柜 EP 通信墙消除

对于一个 `1,024B` 总参数、`64B` 激活参数、`80` 层、`256` 专家（`top-8`）的旗舰模型，在 `BS = 128` 并发 Decode 下，每个 Step 几乎会激活全部 `256` 个专家中的 `~251.7` 个（独特专家激活率 $1 - (248/256)^{128} = 98.3\%$）：

1. **Hopper `H100 NVL8` 的跨机柜 RoCE 死穴**：
   - `1T` 模型在 FP8 下仅权重即达 `1,024 GB`，加上 `L_kv=32,768` 的 KV Cache，至少需要 `EP = 64` 张 H100（跨越 **8 台独立的 NVL8 节点**）。
   - 每层 2 次（Dispatch + Combine）、80 层共 **160 次** All-to-All 必须跨越 `400 Gbps` 机柜间 RoCE/IB 网络，仅跨节点通信暴露延迟即高达 **`31.20 ms/token`**，导致端到端 Decode TPOT 高达 **`37.66 ms`**（仅 `53.11 tok/s/GPU`）。
2. **Blackwell `GB200 NVL72` 与 Vera Rubin `VR200 NVL144` 的超节点内闭环**：
   - 在 `NVFP4` (`0.5 Byte/param`) 下，`1T` 模型权重压缩至 **`512 GB`**，`EP = 64` 完全收敛在**单个 `NVL72` 或半个 `NVL144` 机柜级 NVLink 交换域内**（`cross_rack_ep_required = False`）！
   - 依托 `NVLink-6 (3.6 TB/s)` 与 `CUMEM` Fabric Handles 零拷贝多播，80 层共 160 次 EP All-to-All 总耗时从 H100 的 `31.20 ms` 骤降至 `GB200` 的 **`2.96 ms`** 与 `VR200` 的 **`1.48 ms`**。

---

## 3. 定量实验：三代超节点 `1T MoE` Decode TPOT、`1M` 上下文 PD 配比与 `$/1M Tokens` TCO

由 `experiments/vera_rubin_hbm4_nvl144_roofline/run.py` 测算的三代架构全生命周期对比如下：

| 评估指标 (`1T MoE, 64B Active, BS=128, L_kv=32K MLA`) | **Hopper `H100 NVL8`** | **Blackwell `GB200 NVL72`** | **Vera Rubin `VR200 NVL144`** | `VR200` 相对 `H100` / `GB200` 收益 |
| :--- | :--- | :--- | :--- | :--- |
| **FP8 / FP4 算术强度拐点 (`FLOP/Byte`)** | `590.7` / `590.7` | `562.5` / `1,125.0` | `592.6` / **`1,185.2`** | FP4 拐点翻倍，驱动算子融合 |
| **是否需要跨机柜 RoCE EP (`EP=64`)** | **是 (`8 × NVL8` 跨机柜)** | **否 (单 `NVL72` 机柜闭环)** | **否 (单 `NVL144` 容纳 2 套 `EP=64`)** | 彻底消除跨机柜 EP 尾延迟 |
| **80 层 EP All-to-All 累计耗时 (`ms/step`)** | `31.20 ms` | `2.96 ms` | **`1.48 ms`** | 通信耗时压缩 **`21.1x`** |
| **`1T MoE` Decode TPOT (`ms/token`)** | `37.66 ms` | `4.31 ms` | **`2.28 ms`** | **`16.52x`** vs H100 / **`1.89x`** vs GB200 |
| **单 GPU 吞吐量 (`tokens/s/GPU`)** | `53.11 tok/s` | `464.04 tok/s` | **`877.19 tok/s`** | 单卡产出提升 **`16.52x`** |
| **`1M` 超长上下文 Prefill 耗时 (`8-GPU Equiv.`)** | `142.00 s` | `31.22 s` | **`17.56 s`** | 1M Prefill 提速 **`8.09x`** |
| **`1M` 上下文最优 `Prefill : Decode` 节点配比** | **`4P : 1D`** | **`2P : 1D`** | **`1P : 1D`** | Prefill 算力跃升收敛 PD 配比 |
| **3 年全摊销每百万 Token 成本 (`$/1M tokens`)** | `$12.3263` | `$1.7934` | **`$1.2272`** | 成本下降 **`10.04x`** vs H100 (**`1.46x`** vs GB200) |
