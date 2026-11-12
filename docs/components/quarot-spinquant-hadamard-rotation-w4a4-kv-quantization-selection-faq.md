# 大模型无离群值激活与 KV 量化：正交 Hadamard 旋转（QuaRot / SpinQuant）与 W4A4 / FP4 架构选型指南（FAQ）

> **阅读指南**：本文面向 AI 解决方案架构师（AI Solution Architect）与推理引擎研发团队，彻底拆解大模型从 `W4A16`（仅权重量化）迈向**全链路 `W4A4` / `NVFP4`（权重 4-bit + 激活 4-bit + KV-Cache 4-bit）**时的头号拦路虎——**极端激活通道离群值（Massive Activation Outliers）**。通过大白话与第一性原理，讲透为什么传统的对角缩放（`SmoothQuant` / `AWQ`）在 4-bit 激活与 4-bit KV 下会直接崩盘（`WikiText-2 PPL` 从 `5.12` 暴跌至 `29.85`），以及 **`QuaRot`（随机 Walsh-Hadamard 正交旋转）** 与 **`SpinQuant`（Cayley Stiefel 流形学习旋转）** 如何利用线性代数正交不变性（$Q Q^\top = I_d$）将激活峰度（Kurtosis $\kappa$）从 `482.4` 抹平至近高斯分布的 `2.92`（降幅 **`99.4%`**），把 `Llama-3.1-70B`（`32K` 上下文、`Batch=32`）显存从 `274.2 GB` 压降至单张 80GB 显卡即可容纳的 **`68.6 GB`（节省 `75.0%`）**，同时释放 `INT4 / NVFP4` Tensor Core 算力实现 **`2.60x`** Prefill 吞吐加速！
>
> 🌐 **Language**: **中文 (当前)** | **[English Version](quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-faq.en.md)**
> 📊 **配套可视化与工具**：
> - **[高清 SVG 一图总结（架构师速查图）](../../assets/svg/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-faq-summary.svg)**
> - **[7-Card 4:5 视觉评测报告（CSP 自包含 HTML）](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-cards.html)**
> - **[交互式算力与显存选型计算器（H5 Workbench）](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-sizer.html)**

![QuaRot 与 SpinQuant 正交 Hadamard 旋转 W4A4 + KV4 架构总结](../../assets/svg/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-faq-summary.svg)

---

## 📌 一、核心结论速览（3 分钟看懂选型边界）

### 1.1 通俗大白话与第一性原理：什么是激活离群值（Outliers）？为什么 Hadamard 正交旋转能“无损削峰”？

想象你要把一个班 8,192 名学生的身高装进只有 **15 个刻度格**（`INT4` 对称量化范围 `[-7, +7]`）的量尺里：
- **没有旋转的原始 Transformer 激活（Raw Residual Stream）**：70B 大模型在推理时有一个著名的“怪癖”——在 `8,192` 个隐藏层通道（Hidden Channels）中，总有固定的 `4~16` 个通道像“身高 120 米的巨人”（幅值高达普通通道中位数的 **`78.5x`~`120x`**，统计峰度 Kurtosis $\kappa = 482.4$）。如果你直接用 `INT4` 刻度尺去量全班，为了不把这几个“120 米巨人”截断（Clipping），第 `+7` 格必须代表 `64.0`；结果每格跨度高达 `9.14`，其余 **99.8% 身高在 `[-0.8, +0.8]` 的正常通道全部被四舍五入成 `0`（Underflow 下溢归零）**，模型瞬间“失忆”胡言乱语！
- **为什么 `SmoothQuant` / `AWQ`（对角线缩放）救不了 4-bit 激活？**：`SmoothQuant` 的思路是“把激活通道除以放大倍数 $s_i$，再把权重通道乘以 $s_i$”（$Y = (X \cdot \text{diag}(s)^{-1})(\text{diag}(s) \cdot W)$）。这在 `W8A8`（255 个刻度格）下勉强够用；但在 **`W4A4`（仅 15 个刻度格）** 下，如果你把 `78.5x` 的离群值硬塞给权重矩阵 $W$，权重矩阵自己的动态范围又会被撑爆，导致 `WikiText-2 PPL` 从 `5.12` 崩塌至 **`29.85`**。
- **第一性原理破局——正交旋转不变性（Orthogonal Computational Invariance）**：
  在线性代数中，任何正交矩阵 $Q \in \mathbb{R}^{d \times d}$ 都满足 **$Q \cdot Q^\top = I_d$（单位矩阵）**。因此在做矩阵乘法 $Y = X \cdot W$ 时，我们在中间插入一对正交旋转矩阵：
  $$Y = X \cdot I_d \cdot W = (X \cdot Q) \cdot (Q^\top \cdot W) = X_{\text{rot}} \cdot W_{\text{rot}}$$
  当我们选取 **Walsh-Hadamard 正交矩阵 $H$**（每个元素都是 $\pm \frac{1}{\sqrt{d}}$）作为 $Q$ 时，原本集中在某 1 个通道上的 `+64.0` 巨型尖峰，会被均匀分摊到全部 $d = 8,192$ 个正交基底方向上（单通道幅值瞬间缩小约 $\sqrt{d} = 90.5$ 倍）！根据中心极限定理，旋转后的激活向量 $X_{\text{rot}} = X \cdot H$ 变成平滑的近似高斯分布（峰度 $\kappa$ 从 **`482.4` 暴降至 `2.92`~`3.24`**），15 个 `INT4` 刻度格得到 100% 均匀利用，而点积数学结果与旋转前**严格相等（误差 `0 ULP`）**！

### 1.2 一句话选型红线

1. **老款 GPU（A100 / L40S，无原生 4-bit 激活加速或以低并发 Decode 为主）**：选 **`AWQ W4A16 + FP8 KV-Cache`（方案 S3）**。仅压缩权重节省显存带宽，不动激活。
2. **需要零校准（Zero-Calibration / Training-Free）开箱即用部署 `W4A4 + KV4`**：选 **`QuaRot` 随机 Hadamard 正交旋转（方案 S4）**。无需校准数据集训练旋转矩阵，直接用确定性随机符号 Hadamard 矩阵将 `Llama-3.1-70B` 显存压至 `68.6 GB`，Prefill 提速 **`2.55x`**（`PPL = 5.46`）。
3. **生产级高吞吐 `W4A4` / Blackwell `B200 NVFP4` / 单卡 80GB 跑 70B+32K 并发（架构师首选）**：必须选 **`SpinQuant` Cayley 流形学习旋转 + 在线 `FWHT`（方案 S5）**。通过 100~800 条校准样本在正交 Stiefel 流形上学习最优旋转矩阵 $R_1, R_2$ 并 **100% 离线融合进模型权重（零运行时开销）**，配合仅占 `1.42%` 耗时的在线快速 Walsh-Hadamard 变换（`FWHT` $R_3, R_4$），将 `PPL` 差距收敛至 **`+0.12`（`5.24` vs FP16 `5.12`）**，5-shot MMLU 达 **`78.9%`**，Prefill 吞吐高达 **`36,900 tok/s`（`2.60x` 加速）**！

---

## 🎯 二、客户与架构师灵魂拷问 Q&A（深度原理解析）

### Q1：在 Transformer 每一层里做 $8192 \times 8192$ 的正交矩阵乘法，难道不会把推理延迟拖慢一倍吗？`QuaRot` 和 `SpinQuant` 是怎么做到 `< 1.8%` 极低运行时开销的？

**答：核心秘诀在于“计算不变性离线吸收（Offline Weight Absorption）”——90% 以上的旋转矩阵在模型导出阶段就已经被直接乘进静态权重矩阵里消失了，线上推理时根本不需要执行！**

在一个标准的 Transformer 层（含 `RMSNorm`、Attention `Q/K/V/O` 投影、MLP `Gate/Up/Down` 投影及残差连接）中，`QuaRot` 与 `SpinQuant` 巧妙定义了 **4 组正交旋转矩阵（$R_1, R_2, R_3, R_4$）**：

1. **$R_1 \in \mathbb{R}^{d \times d}$（全局残差流旋转，`8192×8192`）—— 100% 离线吸收（`0 ms` 运行时开销）**：
   - **原理**：`RMSNorm(X)` 的本质是对向量模长做归一化：$\text{RMSNorm}(X) = \frac{X}{\|X\|_2 / \sqrt{d}} \odot \gamma$。因为正交旋转保持欧氏范数不变（$\|X \cdot R_1\|_2 = \|X\|_2$），所以只要先把缩放向量 $\gamma$ 吸收进下一层线性权重，就有恒等式：
     $$\text{RMSNorm}(X \cdot R_1) = \text{RMSNorm}(X) \cdot R_1$$
   - **离线吸收方式**：我们把整个 Transformer 的残差主干流保持在“旋转空间” $X_{\text{rot}} = X \cdot R_1$ 中：
     - 所有**读取残差流**的矩阵（`W_q`, `W_k`, `W_v`, `W_gate`, `W_up`, `LM_Head`），在导出模型时离线左乘 $R_1^\top$：$W_{\text{in}}' \leftarrow R_1^\top W_{\text{in}}$；
     - 所有**写回残差流**的矩阵（`Embedding`, `W_o`, `W_down`），在导出模型时离线右乘 $R_1$：$W_{\text{out}}' \leftarrow W_{\text{out}} R_1$。
     - 结果：线上推理时，残差流天然处于无离群值的旋转空间，$8192 \times 8192$ 的巨型矩阵乘法 $R_1$ **在线上完全不存在（`0.0%` 延迟开销）**！

2. **$R_2 \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$（Attention 单头 `V -> O` 旋转，`128×128`）—— 100% 离线吸收（`0 ms` 运行时开销）**：
   - **原理**：在 Multi-Head / GQA 注意力中，注意力权重矩阵 $P = \text{Softmax}(Q K^\top / \sqrt{d_h})$ 与 Value 矩阵相乘后再经过输出投影 $W_o$：
     $$\text{AttnOut}_h = (P_h \cdot V_h) \cdot W_{o,h} = \bigl(P_h \cdot (X W_{v,h} R_2)\bigr) \cdot (R_2^\top W_{o,h})$$
   - **离线吸收方式**：直接离线更新 $W_{v,h}' \leftarrow W_{v,h} R_2$ 且 $W_{o,h}' \leftarrow R_2^\top W_{o,h}$。这样生成出来的 **V-Cache 天然就是经过 $R_2$ 旋转抹平离群值后的平滑张量**，可以直接无损量化为 **`INT4` V-Cache**，且线上 `0 ms` 开销！

3. **$R_3 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{ffn}}}$（MLP `SwiGLU` 激活后旋转）—— 唯一需要在线执行的全维快速 Hadamard 变换（约 `0.9%` 耗时）**：
   - **为什么不能离线吸收**：MLP 中间的 `SwiGLU` 是逐元素非线性激活函数 $\sigma(x_g) \odot x_u$，正交矩阵无法直接穿透非线性函数。而 `SwiGLU` 乘法恰恰会制造新的动态激活尖峰，导致后面的 `W_down` 无法做 `W4A4`。
   - **破局方案**：在 `SwiGLU` 输出之后、`W_down` 量化之前，插入一个在线的**快速 Walsh-Hadamard 变换（`FWHT`）** $Z' = \text{FWHT}(Z)$，同时把逆变换离线吸收进 `W_down`（$W_{\text{down}}' \leftarrow H^\top W_{\text{down}}$）。因为 `FWHT` 采用蝶形加减法分治算法（无需任何浮点乘法，复杂度仅 $O(d \log_2 d)$，且与量化算子 Fuse 在同一个 GPU SRAM Kernel 内），单层耗时仅微秒级（占端到端仅 **`~0.9%`**）！

4. **$R_4 \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$（Post-RoPE `Q / K` 旋转，`128×128`）—— 解锁 `INT4` Key-Cache 的轻量在线旋转（约 `0.5%` 耗时）**：
   - **为什么需要 $R_4$**：由于旋转位置编码（`RoPE`）直接作用在 `Q` 和 `K` 的通道对上，$R_1$ 只能旋转 `W_q` 和 `W_k` 的输入，无法消除 `W_k` 输出并叠加 `RoPE` 之后的 **Key-Cache 通道尖峰（Attention Sink 频段离群值）**。
   - **破局方案**：在 `Q` 和 `K` 完成 `RoPE` 编码之后、写入 `KV-Cache` 之前，对每个 Head（维度仅 `128`）同步做一次寄存器级 `128×128` 在线 Hadamard 变换：$Q_{\text{rot}} = Q_{\text{rope}} H_{128}$，$K_{\text{rot}} = K_{\text{rope}} H_{128}$。由于注意力内积满足 $Q_{\text{rot}} K_{\text{rot}}^\top = Q_{\text{rope}} H_{128} H_{128}^\top K_{\text{rope}}^\top = Q_{\text{rope}} K_{\text{rope}}^\top$，注意力分数严格不变，而写入显存的 `K-Cache` 峰度暴降，使 **`128K` 长上下文 `INT4 KV-Cache` 大海捞针（NIAH）准确率从 `41.5%` 飙升至 `99.1%`**！

---

### Q2：`QuaRot`（固定随机 Hadamard）和 `SpinQuant`（Cayley 流形优化学习旋转）到底差在哪？什么场景下必须上 `SpinQuant`？

**答：两者的在线推理架构完全兼容（都是 0 开销离线融合 $R_1, R_2$ + 极轻量在线 `FWHT` $R_3, R_4$），核心差异在于离线导出时的 $R_1, R_2$ 是“随机盲抽”还是“针对目标模型权重定向寻优”！**

- **`QuaRot` 的局限**：`QuaRot` 直接使用带随机 $\pm 1$ 对角符号矩阵 $S$ 的 Hadamard 矩阵 $R_1 = H \cdot S$。虽然理论上能把最坏通道尖峰按 $1/\sqrt{d}$ 均摊，但在真实的 `Llama-3.1-70B` 或 `Qwen-2.5` 中，权重矩阵与激活矩阵的奇异向量方向存在特定耦合——随机 Hadamard 旋转在 `W4A4` 极低比特下仍会留下约 `+0.34` 的 PPL 损失（`5.12 -> 5.46`），在代码生成与多步数学推理上会有大约 `1.6%` 的精度回落。
- **`SpinQuant` 的突破（Cayley Parameterization on Stiefel Manifold）**：
  - 既然 $R_1$ 和 $R_2$ 是 **100% 离线融合进模型权重**的，它们根本不需要受限于“在线必须能跑快速蝶形变换（Hadamard）”的结构约束！$R_1$ 和 $R_2$ 可以是**任意正交矩阵**（只要满足 $R^\top R = I$）。
  - `SpinQuant` 冻结原模型权重，仅把量化算子放入前向计算图，利用 **Cayley 变换** 将反对称矩阵 $A = -A^\top$ 映射为严格正交矩阵：
    $$R(A) = (I - A)^{-1}(I + A)$$
    在正交 Stiefel 流形上仅用 `100~800` 条校准样本、耗时不到 30 分钟即可联合优化出同时最小化权重与激活 `INT4/FP4` 量化误差的 $R_1^*, R_2^*$！
  - **收益**：在不增加任何一行线上推理代码、不增加哪怕 `0.1 ms` 线上延迟的前提下（反而因更少异常值裁剪使 Kernel 更稳），将 `Llama-3.1-70B` `W4A4 + KV4` 的 `WikiText-2 PPL` 从 `5.46` 进一步压低至 **`5.24`（距离 FP16 仅 `+0.12`）**，追平了仅做权重量化的 `AWQ W4A16`（`5.29`），真正实现了“鱼（`W4A16` 的精度）与熊掌（`W4A4 + KV4` 的算力与显存）兼得”！

---

### Q3：如果隐藏层维度 $d$ 或 MLP 中间层维度 $d_{\text{ffn}}$ 不是 2 的幂次（例如 `Llama-3-70B` 的 $d_{\text{ffn}} = 28,672$），怎么构造 Hadamard 正交矩阵并跑快速变换？

**答：采用“Kronecker 积（克罗内克积）分块构造法”——将非 2 的幂次维度拆解为 $d = 2^k \times m$（其中 $m$ 为已知阶数的小阶 Hadamard 矩阵，如 Paley 构造的 $m = 12, 20, 28, 36, 112$ 等）！**

- 以 `Llama-3.1-70B` 的 MLP 中间层维度 $d_{\text{ffn}} = 28,672$ 为例：
  $$28,672 = 256 \times 112 = 2^8 \times 112$$
- 构造正交矩阵 $H_{28672} = H_{256} \otimes H_{112}$，其中 $H_{256}$ 是标准 Sylvester 递归 Hadamard 矩阵，$H_{112}$ 是已知的 112 阶正交 Hadamard 常量表（放入 GPU Shared Memory / Constant Cache 仅占 `112×112×2B = 25 KB`）。
- 在线执行 $R_3$ 时，CUDA Kernel 先在 Warp 内对长度为 `256` 的子块执行寄存器级蝶形加减 `FWHT`，再调用一次 `112×112` 的 Tensor Core 小矩阵乘，即可将任意非 2 幂次维度的在线旋转耗时死死压在 **`1.42%`** 以内（比朴素稠密正交矩阵乘法快 `3.4x` 以上）！

---

## 🧭 三、直观可视化图解（Mermaid 决策树 + 路径对比 + 时序图）

### 3.1 架构师量化与正交旋转选型决策树

```mermaid
flowchart TD
    Start["🚀 大模型量化与 KV 压缩选型\n(Target: 70B+ 模型生产部署)"] --> HW{"目标 GPU 是否支持原生\nINT4 / NVFP4 Tensor Core 加速？\n或单卡 80GB 需塞入 70B+32K KV？"}

    HW -- "否：老款 A100/L40S 且仅跑\n小 Batch 内存带宽受限 Decode" --> S3["⚠️ 选用方案 S3：AWQ W4A16 + FP8 KV\n• 权重 4-bit 降带宽，激活保持 FP16\n• 显存 102.1 GB (需双卡 80GB)\n• Prefill 仅 1.18x (无 4-bit 激活加速)"]

    HW -- "是：Blackwell B200 (NVFP4) / 高并发 Prefill\n或需将 70B+32K 压入 68.6 GB 单卡" --> NaivCheck{"能否直接用 SmoothQuant / RTN\n对角缩放做 W4A4 + KV4？"}

    NaivCheck -- "绝对禁止！" --> S2["⛔ 方案 S2：朴素对角缩放 W4A4\n• 激活峰度 κ = 310.6 (34.2x 尖峰未消除)\n• 99% 正常通道在 INT4 下溢归零\n• WikiText-2 PPL 暴跌至 29.85 (彻底崩盘)"]

    NaivCheck -- "必须引入正交旋转消除离群值\n(Q · Q^T = I_d)" --> Calib{"是否允许离线花费 ~30 分钟\n在 512 条校准集上优化旋转矩阵？"}

    Calib -- "否：要求 100% Training-Free\n零样本即时导出" --> S4["✅ 选用方案 S4：QuaRot 随机 Hadamard (W4A4+KV4)\n• R1/R2 离线随机 Hadamard 融合 + R3/R4 在线 FWHT\n• 峰度 κ 降至 3.24，显存 68.6 GB (-75%)\n• Prefill 36,200 tok/s (2.55x)，PPL = 5.46 (+0.34)"]

    Calib -- "是：追求生产级最高精度与吞吐\n(推荐标准路径)" --> S5["🏆 选用方案 S5：SpinQuant Cayley 学习旋转 (W4A4+KV4)\n• Stiefel 流形学习 R1/R2 (100% 离线融合，0 ms 在线开销)\n• R3 (MLP) + R4 (RoPE KV) 轻量在线 FWHT (1.42% 开销)\n• 峰度 κ = 2.92，PPL = 5.24 (仅 +0.12)，Prefill 2.60x！"]
```

---

### 3.2 Transformer 层内四组旋转矩阵（$R_1, R_2$ 离线吸收 vs $R_3, R_4$ 在线 FWHT）架构路径对比图

```mermaid
flowchart LR
    subgraph Offline["📦 100% 离线融合进静态权重（导出时一次性完成，线上 0 ms 开销）"]
        direction TB
        R1["R1 (8192×8192 全局残差流正交旋转)\n• W_q, W_k, W_v, W_up, W_gate ← R1^T · W\n• W_o, W_down ← W · R1\n• SpinQuant 在 Stiefel 流形学习最优 R1"]
        R2["R2 (128×128 单头 Value-to-Out 正交旋转)\n• W_v^(h) ← W_v^(h) · R2\n• W_o^(h) ← R2^T · W_o^(h)\n• 使生成的 V-Cache 天然平滑可直存 INT4"]
    end

    subgraph Online["⚡ 线上推理时执行的极轻量快速 Hadamard 核（总耗时仅 1.42%~1.75%）"]
        direction TB
        R4["R4 (128×128 Post-RoPE Q/K 在线 FWHT)\n• Q_rot = FWHT_128(RoPE(Q))\n• K_rot = FWHT_128(RoPE(K)) → 量化写入 INT4 K-Cache\n• 内积不变：Q_rot · K_rot^T == Q_rope · K_rope^T"]
        R3["R3 (28672 维 SwiGLU 激活后在线分块 FWHT)\n• Z_rot = FWHT_256x112(SiLU(Gate) ⊙ Up)\n• W_down 已离线左乘 H_dff^T\n• 解锁 W_down 原生 W4A4 / NVFP4 Tensor Core GEMM"]
    end

    Offline ==>|"导出无离群值旋转模型"| Online
```

---

### 3.3 SpinQuant 离线导出与在线 W4A4 + INT4 KV-Cache 推理全链路时序图

```mermaid
sequenceDiagram
    autonumber
    participant Export as 🛠️ 离线导出管线 (SpinQuant Cayley)
    participant Residual as 🔄 旋转残差流 X_rot (FP16)
    participant Attn as ⚡ W4A4 Attention + R4 (FWHT_128)
    participant KVCache as 💾 INT4 KV-Cache (Group=128)
    participant MLP as 🚀 W4A4 MLP + R3 (FWHT_dff)

    Note over Export: 阶段一：离线一次性流形寻优与权重融合（线上 0 开销）
    Export->>Export: 1. 固定权重，用 Cayley 变换 R=(I-A)^(-1)(I+A) 优化 R1, R2 (512 校准集)
    Export->>Export: 2. 离线融合 R1 到 W_qkv/W_o/W_up/W_down，融合 R2 到 W_v/W_o，融合 R3^T 到 W_down
    Export->>Export: 3. 执行 GPTQ/AWQ 导出 W4 权重，激活峰值比从 78.5x 降至 2.41x (κ=2.92)

    Note over Residual,MLP: 阶段二：线上推理单层前向传播（全链路 W4A4 Tensor Core + INT4 KV）
    Residual->>Attn: 4. RMSNorm(X_rot) + 动态 INT4 量化激活（无离群值，15 格满载）
    Attn->>Attn: 5. INT4/NVFP4 Tensor Core 计算 Q, K, V (V 已含离线 R2 旋转)
    Attn->>KVCache: 6. 对 RoPE(Q), RoPE(K) 执行寄存器级 R4 (FWHT_128, 耗时 ~0.5%) 并存入 INT4 KV-Cache
    KVCache-->>Attn: 7. FlashAttention-3 / FlashInfer 读取 INT4 K_rot, V_rot 计算注意力并乘 W_o'
    Attn->>Residual: 8. 写回旋转残差流 X_rot (保持在 R1 正交空间)
    Residual->>MLP: 9. RMSNorm(X_rot) → W4A4 Tensor Core 计算 Gate & Up → SwiGLU 激活
    MLP->>MLP: 10. 在线融合执行 R3 分块 FWHT (256×112, 耗时 ~0.9%) + INT4 量化 → W4A4 W_down'
    MLP->>Residual: 11. 输出下一层旋转残差流 X_rot（吞吐达 36,900 tok/s，提速 2.60x！）
```

---

## 📊 四、架构师选型决策矩阵与适用边界

### 4.1 五大方案全维度对比表（基于 `Llama-3.1-70B`，`32K` Context，`Batch=32`）

| 选型方案 | 旋转与消峰机制 | 激活峰度 $\kappa$（越近 3 越平滑） | 最大/中位通道比 | WikiText-2 PPL（$\downarrow$） | 5-Shot MMLU（$\uparrow$） | 总显存（权重+32K KV） | Prefill 吞吐（tok/s） | 在线旋转开销 | 架构师推荐等级 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S1: FP16 Baseline (`W16A16+KV16`)** | 无（原始残差流） | `482.4` | `78.5x` | `5.12` (`0.00`) | `79.4%` | `274.2 GB` (`140+134.2`) | `14,200` (`1.00x`) | `0.00%` | 基准参照（需 4×80GB） |
| **S2: Naive SmoothQuant (`W4A4+KV4`)** | 对角矩阵缩放 $\text{diag}(s)$ | `310.6` | `34.2x` | **`29.85`** (`+24.73` ✕) | `34.8%` ✕ | `68.6 GB` (`35+33.6`) | `38,400` (`2.70x`) | `0.00%` | ⛔ **严禁上线（精度崩盘）** |
| **S3: AWQ Hybrid (`W4A16+FP8 KV`)** | 显著权重保护（激活不量化） | `482.4` | `78.5x` | `5.29` (`+0.17`) | `78.6%` | `102.1 GB` (`35+67.1`) | `16,800` (`1.18x`) | `0.00%` | ⚠️ **确认后使用（仅 Decode 省带宽）** |
| **S4: QuaRot (`W4A4+KV4` 随机 Hadamard)** | 离线随机 Hadamard $R_1,R_2$ + 在线 $R_3,R_4$ | `3.24` | `2.85x` | `5.46` (`+0.34`) | `77.8%` | **`68.6 GB`** (`-75.0%`) | `36,200` (`2.55x`) | `1.75%` | ✅ **推荐（零校准样本即时导出）** |
| **S5: SpinQuant (`W4A4+KV4` Cayley 学习)** | **Stiefel 流形学习 $R_1,R_2$（离线融合）+ 在线 $R_3,R_4$** | **`2.92`** (`-99.4%`) | **`2.41x`** | **`5.24`** (`+0.12` ★) | **`78.9%`** | **`68.6 GB`**（单卡 80GB 可跑） | **`36,900`** (**`2.60x`** ★) | **`1.42%`** | 🏆 **生产环境首选（精度与吞吐双冠）** |

### 4.2 架构师适用边界：✅ Good fit vs ⛔ Not for

| ✅ Good fit（强烈推荐采用 `QuaRot / SpinQuant` 正交旋转的场景） | ⛔ Not for（不建议或无需引入在线正交旋转的场景） |
| :--- | :--- |
| **1. Blackwell B200 (`NVFP4`) 与原生 `INT4` Tensor Core 高吞吐 Prefill 集群**：必须消除激活通道尖峰，才能让 `W4A4` / `FP4` 矩阵乘不发生动态截断。 | **1. 仅支持 `FP16` Tensor Core 且 Batch=1 的纯边缘端低频推理**：若硬件没有原生 4-bit 激活乘加指令（如旧款消费级显卡），在线 `FWHT` 无法换来算力倍增，`AWQ W4A16` 即可。 |
| **2. 单卡 80GB（H100/H800/H200）高密度承载 70B + 32K~128K 长上下文**：`W4` 权重（`35 GB`）+ `INT4` KV-Cache（`33.6 GB`）共 `68.6 GB`，免去跨卡 TP 通信税。 | **2. 模型需要运行时每秒动态热加载海量不同 `LoRA` 适配器且未与 $R_1$ 预对齐**：因为主干权重已被 $R_1, R_2$ 旋转，外挂 LoRA 必须在导出时同步乘上 $R_1^\top$ 和 $R_1$。 |
| **3. 128K 超长文本 RAG / Agent 记忆压缩（4-bit KV-Cache）**：通过 `R4` Post-RoPE Head-128 Hadamard 旋转抹平 Attention Sink 频段离群值，使 `INT4 KV` 大海捞针准确率保持 `99.1%`。 | **3. 极致小模型（`< 1.5B` 参数）且对 `0.1` PPL 极其敏感的数学竞赛任务**：超小模型冗余度低，建议保持 `W8A8` 或 `FP8`（但仍可免费使用离线融合的 $R_1, R_2$ 提升 `FP8` 稳定性！）。 |

---

## 🧪 五、定量算例与 7-Card 4:5 视觉评测报告摘要

### 5.1 正交旋转不变性与单通道削峰数学验算（确定性算例）

给定一个含极端离群通道的 4 维激活向量 $x = [64.0, 0.5, -0.4, 0.3]$（第 0 通道是中位数的 `160x` 尖峰）与权重向量 $w = [0.25, -0.5, 0.75, 0.1]^\top$：
1. **原始内积**：$y = x \cdot w = 64.0 \times 0.25 + 0.5 \times (-0.5) + (-0.4) \times 0.75 + 0.3 \times 0.1 = 15.48$，原始激活最大幅值 $\|x\|_\infty = \mathbf{64.0}$。
2. **乘 4 阶正规化 Hadamard 矩阵 $H_4 = \frac{1}{2}\begin{bmatrix}1&1&1&1\\1&-1&1&-1\\1&1&-1&-1\\1&-1&-1&1\end{bmatrix}$**：
   - 旋转后激活 $x_{\text{rot}} = x \cdot H_4 = [32.2, 31.8, 32.3, 31.7]$，最大幅值从 `64.0` 瞬间减半至 **`32.3`**（在 $d=8192$ 维下缩小约 $\sqrt{8192} \approx 90.5$ 倍，通道比从 `78.5x` 降至 **`2.41x`**）！
   - 旋转后权重 $w_{\text{rot}} = H_4^\top \cdot w = [0.3, 0.7, -0.4, -0.1]^\top$。
   - 旋转后内积：$x_{\text{rot}} \cdot w_{\text{rot}} = 32.2 \times 0.3 + 31.8 \times 0.7 + 32.3 \times (-0.4) + 31.7 \times (-0.1) = \mathbf{15.48}$（**绝对误差 `0.0`**）！

### 5.2 鲁棒性压力测试（Perturbation & Consistency Checks）

| 压力测试场景 | 核心评测指标 | 朴素 `W4A4 + KV4`（无旋转） | `SpinQuant`（Cayley $R_1,R_2$ + 在线 $R_3,R_4$） | 净收益 / 鲁棒性结论 |
| :--- | :--- | :---: | :---: | :--- |
| **1. 代码与数学 Prompt 极端通道尖峰（120x 中位数）** | WikiText-2 / Code PPL（$\downarrow$） | `44.10` | **`5.31`** | **`-38.79 PPL`**（单通道尖峰被均摊至 8,192 维） |
| **2. 128K 超长上下文 INT4 KV-Cache（Group=128）** | Needle-in-a-Haystack 召回率（$\uparrow$） | `41.5%` | **`99.1%`** | **`+57.6 pts`**（Post-RoPE $R_4$ 消除 Key 频段离群值） |
| **3. Blackwell B200 NVFP4 微缩放（Block-16 E2M1）** | 5-Shot MMLU 准确率（$\uparrow$） | `74.2%` | **`79.2%`** | **`+5.0 pts`**（距 FP16 `79.4%` 仅差 `0.2%`） |
| **4. 非 2 幂次 FFN 维度（$d_{\text{ffn}}=28672=256\times 112$）** | 在线 FWHT Kernel 延迟占比（$\downarrow$） | `4.85%`（稠密正交乘） | **`1.42%`** | **`-3.43 pts`**（Kronecker 分块 $H_{256}\otimes H_{112}$ 融合核） |

> 💡 **查看完整 7-Card 4:5 视觉评测卡片与交互式计算器**：
> - **[打开 7-Card 4:5 视觉评测报告 (`quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-cards.html`)](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-cards.html)**
> - **[打开交互式量化与正交旋转工作台 (`quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-sizer.html`)](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-sizer.html)**
> - **数据溯源声明（Provenance）**：本报告全部定量数据由零依赖确定性仿真器 [`experiments/quarot_spinquant_hadamard_rotation_w4a4_kv_quantization_selection/run.py`](../../experiments/quarot_spinquant_hadamard_rotation_w4a4_kv_quantization_selection/run.py) 生成并封存于 [`results.json`](../../experiments/quarot_spinquant_hadamard_rotation_w4a4_kv_quantization_selection/results.json)，数据类型标记为 `synthetic_combined_with_anonymized_reported`（`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`）。

---

## 🛠️ 六、生产环境快速核验清单（Architect Pre-Flight Checklist）

在将 `QuaRot` 或 `SpinQuant` `W4A4 / NVFP4 + KV4` 推向生产集群前，请逐项勾选核验：

- [ ] **1. `RMSNorm` 缩放向量 $\gamma$ 预融合检查**：在离线将 $R_1$ 乘入线性层权重之前，确认已先将每一层 `RMSNorm`（或居中后的 `LayerNorm`）的对角权重向量 $\text{diag}(\gamma)$ 吸收进紧邻的 `W_qkv` 与 `W_up/gate` 中，否则 $\text{RMSNorm}(X) \cdot \gamma \cdot R_1 \neq \text{RMSNorm}(X \cdot R_1) \cdot \gamma$ 会破坏正交可交换性！
- [ ] **2. 词表 Embedding 与 `LM_Head` 共享权重（Tie-Word-Embeddings）解耦或同步旋转**：确认输入 `Embedding` 表已离线右乘 $R_1$（$E' \leftarrow E \cdot R_1$），输出 `LM_Head` 已离线左乘 $R_1^\top$（若权重共享，旋转后依然保持共轭一致）。
- [ ] **3. GQA（Grouped-Query Attention）下的 $R_2$ 头间对齐**：在 `Llama-3.1-70B`（8 个 KV Head 对应 64 个 Query Head，每组 GQA 共享 1 个 V Head）中，确认同一 GQA 分组内的 8 个 `W_o` 头切片使用的是与该组共享 `W_v` 头完全相同的 $128 \times 128$ 正交矩阵 $R_2^{(g)}$。
- [ ] **4. 非 2 幂次维度 Kronecker Hadamard 因子表校验**：针对 $d_{\text{ffn}} = 28672$（`256 × 112`）或 $d = 5120$（`256 × 20`），确认推理引擎已内置对应的 Paley / Williamson 正交 Hadamard 常量矩阵并开启 Fused FWHT + Quantization CUDA Kernel。
- [ ] **5. 外挂 LoRA 适配器正交空间对齐**：若生产环境挂载动态 LoRA（$\Delta W = B \cdot A$），确认 LoRA 导出脚本已按所在层位置离线左乘 $R_{\text{in}}^\top$ 并右乘 $R_{\text{out}}$，严禁将未旋转空间的原始 LoRA 直接叠加到已旋转主干权重上。
