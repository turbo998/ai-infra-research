# FlashInfer Rubin (`sm_107a`) BF16-State GDN Decode uGPU 局域性域切分、Context-Parallel KCP 分块训练规划、Rubin MLA/Kimi-K3 PDL 门控与 TileLang Ascend `Nd2NzGeometry`/PTO 向量本地缓冲/CUDA Cast 舍入护栏深度解析

> **语言 / Language**: [中文](flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1.md) | [English](flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1.en.md)  
> **Phase**: 69 (`priority: 78`, Track A — GPU Kernel & Quantization)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `author-benchmark` (作者 VR200 `sm_107a` / GB300 / B300 / B200 实测报告) + `own-offline` (本项目确定性标准库仿真 `experiments/flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1/run.py`)

---

## 1. 架构背景与核心工程问题

在 NVIDIA Rubin (`sm_107a` VR200)、Blackwell Ultra (`sm_103a` GB300/B300) 与华为昇腾 Ascend NPU 的最新算子栈演进中，硬件代际跃迁带来了四个直接影响系统吞吐与正确性的核心工程问题：

1. **Rubin (`sm_107a` VR200) 双 Die (`uGPU`) 内存局域性与指令带宽失衡 (`flashinfer-ai/flashinfer@6fc0f52beae0`, `#6604`)**：VR200 拥有 `19.15 TB/s` HBM 峰值带宽（约为 B200 的 `2.5x`），但全芯片 `212` 个 SM 的指令发射速率仅为 B200 的 `~1.8x`，且物理上划分为 2 个各含 `106` 个 SM 的 `uGPU` 局域性域（Locality Domains）。为什么移植自 B200 的通用 Gated Delta Net (GDN) BF16-state Decode 算子在 VR200 上仅能达到 `39%–63%` 带宽利用率？如何通过指令精简与 `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN` 物理绑定突破 `88%` 峰值带宽？
2. **Rubin MLA NVFP4 Decode 运行时 PDL 开关 (`flashinfer-ai/flashinfer@513c4ae68a5a`, `#7004`) 与 Kimi-K3 `attn_res` 持久化写回 (`flashinfer-ai/flashinfer@4bfe0963180d`, `#7014`)**：为什么在 `cake_rubin_mla_nvfp4_decode` 中静态开启 Programmatic Dependent Launch (PDL) 会在批大小跨入第二波（2nd wave）Cluster 时造成性能退化？
3. **Context-Parallel (CP) 变长打包 KDA 分块训练规划 (`flashinfer-ai/flashinfer@4c7318173bac`, `#7002`)**：在跨 CP Rank 切分变长打包序列时，如何通过 `"wave_fill1"` 段数规划 (`kcp_plan`) 与 FP32 仿射状态摘要（Affine State Summary）消除串行前缀依赖？
4. **TileLang Ascend 部分 UB-to-L1 NZ 拷贝几何校验、PTO `SimtVF` 向量本地缓冲标量化与 CUDA 随机舍入类型护栏 (`tile-ai/tilelang@59dc2b6763b8` `#3530`, `@a2679e5c6d75` `#3523`, `@e84ce3de8ab7` `#3350`)**：如何为 Ascend 部分分块 `copy_ub_to_l1` 精确推导源/目的跨距与脉冲间隙？为何 `float32 -> float8_e8m0fnu` 必须被排除在 CUDA `rs` 随机舍入向量化路径之外？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 FlashInfer Rubin (`sm_107a`) BF16-State GDN Decode 与 `uGPU` 局域性域切分 (`#6604`)

在 `flashinfer/gdn_kernels/gdn_decode_bf16_state_rubin.py` 与 `flashinfer/gdn_decode.py` 中，FlashInfer 针对 VR200 (`sm_107a`, `212` SMs = `2 × 106` SMs/die, `19.15 TB/s`) 进行了指令路径与物理内存拓扑的双重重构：

- **指令路径瘦身（Warp 指令数削减 `~38%`）**：
  - **同构 Copy-Compute 分区消除 CTA 屏障**：每个 $128 \times 128$ 的 BF16 状态矩阵拆分为两个 $64 \times 128$ 半区（每个 CTA 负责 `64` 行，即 `16 KB` SMEM）。`cp.async` 整块加载与计算均令 Warp 组 $g \in [0, 7]$ 处理行 $g, g+8, \dots, g+56$，由于每个线程仅消费自身异步写入的共享内存行，彻底消除了加载与计算之间的 `bar.sync`。
  - **5-Shuffle 转置归约与 `FFMA2` 打包**：将 16-lane Q/K 向量打包进单线程 8 个 `f32x2` 寄存器对并应用 3-shuffle 蝶形全归约完成 L2 归一化；随后通过 5 次寄存器配对交换（5-shuffle transpose-reduce，替代原有的 16 次跨通道广播循环）直接在线性累加器上完成状态更新，并使用 `FFMA2` 双通道乘加。
  - **L2 `evict_last` 前瞻预取**：在发射本块写回前，使用 `prefetch.global.L2::evict_last` 预取后续第
    $$
    \text{pf\_dist} = \left\lfloor \frac{\text{num\_SMs} \times 25}{3} \right\rfloor = \left\lfloor \frac{212 \times 25}{3} \right\rfloor = 1766 \text{ tiles}
    $$
    （对应每 SM 驻留 7 个 CTA 时约 `1.2` 代驻留周期）的状态首缓存行。
- **`uGPU` 双 Die 物理隔离池 (`ugpu_state_pool`)**：
  - 普通 `cudaMalloc` 按页交错分布在两个 Die 上，导致 ~50% 的访问跨 Die 互联（实测纯拷贝带宽上限从 Die 本地的 `90%` 跌至 `68%`）。
  - `ugpu_state_pool(pool_size, HV)` 通过 `cuMemCreate` 指定 `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`（`id = 0` 与 `id = 1`），分配形状为 `[pool_size, HV, 2, V // 2, K]` 的状态池，使 `half = 0`（行 `0..63`）物理落盘在 Die 0 HBM，`half = 1`（行 `64..127`）落盘在 Die 1 HBM，并在各自的 `106`-SM Green Context 流上执行。
  - **调度阈值契约**：`backend="auto"` 在 SM107 上当 $B \times H_V \ge 512$ (`_AUTO_MIN_STATES`) 时自动启用 `rubin` 内核；对于 `ugpu` 池，当 $B \times H_V < 8192$ (`_UGPU_EAGER_MIN_STATES`) 且处于非捕获 Eager 模式时，在当前流上背靠背启动两个半区以省去約 `5 us` 的跨流事件 fork/join 开销；当 $B \times H_V \ge 8192$ 或处于 CUDA Graph 捕获中时，并行 fork/join 两个 Die 的 Green Context 流。
- **VR200 实测性能 (`author-benchmark`, `HV=64, HQ=HK=16, K=V=128`)**：
  - `B=32`: Generic `18.0 us` (`39%`) $\to$ `rubin` `13.2 us` (`53%`, `1.36x`) $\to$ `rubin_ugpu` **`12.3 us` (`57%`, `1.46x`)**
  - `B=512`: Generic `187.0 us` (`61%`) $\to$ `rubin` `163.0 us` (`69%`, `1.15x`) $\to$ `rubin_ugpu` **`132.6 us` (`85%`, `1.41x`)**
  - `B=2048`: Generic `726.7 us` (`62%`) $\to$ `rubin` `640.2 us` (`71%`, `1.14x`) $\to$ `rubin_ugpu` **`516.5 us` (`88%`, `1.41x`)**
  - `B=8192`: Generic `2887.0 us` (`63%`) $\to$ `rubin` `2549.8 us` (`71%`, `1.13x`) $\to$ `rubin_ugpu` **`2059.8 us` (`88%`, `1.40x`)**

### 2.2 Rubin MLA NVFP4 Decode 运行时 PDL 门控 (`#7004`) 与 Kimi-K3 持久化写回 (`#7014`)

1. **Rubin MLA NVFP4 Decode `pdl_for(plan)` (`#7004`)**：
   - 在 `flashinfer/experimental/rubin_mla_nvfp4_decode/api.py` 中，单波可同时驻留的 Cluster 单元数由 `UNITS_PAIRS_BY_SM_COUNT = {208: (50, 4), 212: (46, 14)}` 给出（即 208-SM 为 $50+4=54$ 个单元，212-SM 为 $46+14=60$ 个单元）。
   - 若 `plan.items > plan.units`（工作项溢出至第 2 波 Cluster），开启 PDL 会使后继 Merge Kernel 的 CTA 在第 1 波 Decode CTA 发出 `griddepcontrol.launch_dependents` 后立即抢占 SM，反而挤占第 2 波 Decode Cluster 的槽位。因此运行时门控严格定义为：
     $$
     \operatorname{pdl\_for}(\text{plan}) = (\text{plan.max\_splits} > 1) \land (\text{plan.items} \le \text{plan.units})
     $$
2. **Kimi-K3 `attn_res` 快照写回与 SM103 Held-Release (`#7014`)**：
   - 在 `flashinfer/experimental/kimi_k3_attn_res/api.py` 中，新增深度 2 `K=4` 快照写回内核 `persistent:k4_nc3_d2_f111000000000000_write`（固定于 `WRITE_PERSISTENT_KEY_CELLS` 查表），并在 `sm_103a` (GB300/B300) 上将 `K=5, M >= 2048` 路由至深度 3 `nc3` held-release 算子（GB300 `M=4096, K=5` 从 `85.248 us` 降至 `83.920 us` `1.0160x`；B300 `M=4096, K=4 write` 从 `87.649 us` 降至 `85.473 us` `1.0255x`）。

### 2.3 Context-Parallel (CP) 变长打包 KDA 分块训练规划 (`#7002`)

在 `flashinfer/experimental/kcp/cake_kcp.py` 与 `cake_kcp_plan.py` 中，FlashInfer 实现了面向变长打包序列的 Context-Parallel KDA 分块递推内核：
- **三阶段仿射状态归约**：每个局部序列段将 chunk 递推归纳为 FP32 仿射转移算子 $(A_{\text{seg}}, B_{\text{seg}})$，跨 CP 进程组执行 `all_gather_into_tensor` 后按因果顺序复合初始状态，再写回全精度 `O` 与终态 `ht`，并支持 `contiguous`、`zigzag` 与 `zigzag_per_document` 三种切分以及宽度为 4 的因果 `conv1d + SiLU` 边界 Halo 交换。
- **`"wave_fill1"` 段数选择器 (`kcp_plan`)**：
  - 当单 Rank 仅含 $n_{\text{frag}} = 1$ 个跨 Rank 片段且 $0 < NC_{\max} < 257$ 时，按单波填满 SM 选取段数：
    $$
    P = \max\!\left(P_{\text{chain}},\; \min\!\left(\left\lfloor \frac{\text{SM}}{H} \right\rfloor,\; \min(64, NC_{\max})\right)\right), \quad P_{\text{chain}} = \left\lceil \frac{NC_{\max}}{16} \right\rceil
    $$
  - 当 $n_{\text{frag}} \le 3$ 且 $NC_{\max} \ge 257$ 时，最小化波次步进与状态复合联合代价 $\lceil n_{\text{frag}} P H / \text{SM} \rceil \cdot \lceil NC_{\max} / P \rceil \cdot t_{\text{walk}} + (P - 1) t_{\text{compose}}$。在 GB300 (`sm_103a`) 144 组打包配置上，相比 FLA KCP 提速 **`1.3058x–2.4679x`**，相比 Cake Head-CP 提速 **`1.6184x–2.3319x`**。

### 2.4 TileLang Ascend `Nd2NzGeometry`、PTO 向量本地缓冲与 CUDA Cast 舍入护栏 (`#3530`, `#3523`, `#3350`)

1. **Ascend 部分 UB-to-L1 NZ 拷贝 (`Nd2NzGeometry`, `#3530`)**：
   - 在 `src/tl_templates/ascend/common.h` 与 `src/ascend/transform/lower_Tile_op.cc` 中，新增支持显式源/目的行跨距（`src_pitch`, `dst_pitch`）的 `copy_ub_to_l1` 重载。设每个 32B C0 分形块容纳 $C_0 = 32 / \text{dst\_bytes}$ 个元素，则分形列脉冲数为 $\text{Bursts} = \text{cols} / C_0$，源/目的间隙为 $\text{src\_gap} = \text{src\_pitch} - \text{rows}$、$\text{dst\_gap} = \text{dst\_pitch} - \text{rows}$，覆盖字节跨度为：
     $$
     \operatorname{SpanBytes}(\text{pitch}) = \bigl((\text{Bursts} - 1) \cdot \text{pitch} + \text{rows}\bigr) \times 32
     $$
     并强制校验 $1 \le \text{rows} \le 65535$ 且 $1 \le \text{Bursts} \le 4095$。
2. **PTO `T.SimtVF` 向量本地缓冲标量化 (`#3523`)**：
   - 在 `tilelang/Carver/arch/driver/pto_Pass.py` 中，将 `T.alloc_local((N,), "float32xL")` 重写为标量寻址的 `pto.alloc_buffer((N * L,), pto.f32)`，并将向量读写拆解为标量偏移 `slot * L + lane`，避免后端生成非法的向量指针解引用。
3. **CUDA `cast_round="rs"` 随机舍入前置校验 (`#3350`)**：
   - 在 `src/transform/lower_vectorize.cc` 中，`CastRoundIsSupported` 严禁将非 `float32` 源类型、无 `rbits` 参数或目标为 `float8_e8m0fnu`（E8M0 指数格式硬件无条件向 $+\infty$ 舍入，不支持随机舍入）的 `T.Cast` 送入向量化 PTX `cvt.rs` 路径，在编译期直接抛出诊断信息。

---

## 3. 本项目确定性离线实验验证 (`own-offline`)

运行 `python experiments/flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1/run.py --check`，产出 `summary.json`，核心验证结果如下表：

| 验证维度 | 指标 / 场景 | 基线 / 负对照 | 优化 / 护栏后结果 | 证据等级 |
|---|---|---|---|---|
| **Rubin GDN Decode + uGPU (`#6604`)** | VR200 (`212` SMs, `19.15 TB/s`) `B=32..8192` | Generic B200 内核：`B=2048` 延迟 `726.7 us` (`62%` 带宽) | `pf_dist=1766` tiles；`rubin` `640.2 us` (`71%`)；`rubin_ugpu` **`516.5 us` (`88%`, `1.41x`)** | `author-benchmark` + `own-offline` |
| **Rubin MLA PDL 与 Kimi-K3 (`#7004/#7014`)** | 208-SM `items=54` vs `55`；B300 `M=4096, K=4` write | `items=55` 开 PDL 抢占第 2 波；Kimi-K3 旧写回 `87.649 us` | `items=54 -> PDL=True`，`items=55 -> PDL=False`；B300 新写回 **`85.473 us` (`1.0255x`)** | `author-benchmark` + `own-offline` |
| **CP KDA `"wave_fill1"` (`#7002`)** | GB300 (`148` SMs) `n_frag=1, NC=64, H=16` | 固定 `P_chain = 4`（仅占 `64/148` SM） | `kcp_plan` 扩展至 `P=9` 填满单波 SM；144 配置较 FLA KCP 快 **`1.31x–2.47x`** | `author-benchmark` + `own-offline` |
| **TileLang Ascend / PTO / Cast (`#3530/#3523/#3350`)** | `16x64` FP16 子块拷贝；`float32x4` 2 槽；`f32->e8m0` `rs` | 全块拷贝覆盖邻列；向量指针报错；`e8m0` 静默错误 | `dst_span=6656 B` (`rows>65535` 拒绝)；标量缓冲 `8` 元素；`e8m0`/`f16` `rs` 编译期拦截 | `implementation` + `own-offline` |

---

## 4. 面向解决方案架构师的练习与可证伪问题

1. **架构计算题（VR200 uGPU 跨 Die 流量与延迟预算）**：给定 `B=2048, HV=64, K=V=128`，BF16 状态矩阵总字节数为 $2048 \times 64 \times 128 \times 128 \times 2 \text{ B} = 4.295 \text{ GB}$。单步 Decode 需读取并写回全部状态（总 HBM 流量 $8.590 \text{ GB}$）。请验证在 `19.15 TB/s` 峰值带宽下，理论最小传输时间与 `rubin_ugpu` 实测 `516.5 us` 对应的 `88%` 有效带宽利用率。
2. **可证伪假设**：在 `cake_rubin_mla_nvfp4_decode` 中，当 `plan.items > plan.units`（例如 208-SM 上 `items = 64 > 54`）时若强制开启 `enable_pdl=True`，Nsight Compute 时间线将观测到 Merge CTA 在第 1 波 54 个 Decode Cluster 发出 `launch_dependents` 后立刻抢占 SM，导致剩余 10 个第 2 波 Decode Cluster 被推迟至 Merge 结束后才完成。
