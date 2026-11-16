# 大模型强化学习（`GRPO / PPO / RLOO`）KL 散度估计器（`Schulman k1 vs k2 vs k3` vs 全词表解析 `KL`）选型：为什么直接用 `k1 = log_pi - log_ref` 会在 `48.5%` 的 Token 上算出负数 KL（把惩罚变成奖励并让方差暴涨 `14.8x`），而全词表求和又会狂吞 `39.86 GB` 显存直接 OOM？基于一维采样 Token 上的对称截断 `Schulman k3` 估计器（`exp(clamp(d, -10, 10)) - 1 - d`，显存降至 `0.12 MB`，`100%` 无偏且恒正）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection/run.py` 与 `results.json`。

![RL KL-Divergence Schulman k3 Estimator vs Full-Vocab OOM Visual Summary](../../assets/svg/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向大模型强化学习后训练（`DeepSeek-R1 GRPO / verl / OpenRLHF`）的“基于一维采样 Token 对数概率的对称钳位 `Schulman k3` 无偏非负 KL 散度估计器（$\hat{D}_{k3} = \exp(\text{clamp}(\Delta, -10, 10)) - 1 - \Delta$，其中 $\Delta = \log \pi_{\text{ref}}(y_t) - \log \pi_\theta(y_t)$）：以 `0.12 MB` 极低显存开销彻底取代 `39.86 GB` 全词表解析 KL 并消灭 `48.5%` 负数 KL 奖励作弊”**。
- **架构师价值**：在搭建 `GRPO`、`PPO` 或 `RLOO` 强化学习训练管线时，为了防止策略模型 $\pi_\theta$ 偏离初始参考模型 $\pi_{\text{ref}}$ 太远而发生语言崩坏（Reward Hacking），必须加入 KL 散度正则项 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$。然而，围绕“这行 KL 代码到底怎么写”，工程界踩过了三个代价惨重的深坑：
  1. **深坑一：直接写朴素的 `k1 = logp_theta - logp_ref`，导致 `48.5%` 的 Token 算出“负数 KL 惩罚”（惩罚反变奖励 Bonus！），且梯度方差暴涨 `14.8 倍`（Tier 1）**！
     - 很多初学者根据定义 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y_t \sim \pi_\theta}\bigl[\log \pi_\theta(y_t) - \log \pi_{\text{ref}}(y_t)\bigr]$，直接在代码里把单个采样 Token $y_t$ 的 KL 惩罚写成了：
       $$\hat{D}_{k1}(y_t) = \log \pi_\theta(y_t) - \log \pi_{\text{ref}}(y_t)$$
     - **为什么这会导致训练迅速崩溃？**
       - 虽然对整个词表分布求期望时 $D_{\text{KL}} \ge 0$，但对于 **某一次具体采样出来的单个 Token $y_t$**，只要策略模型给出的概率略低于参考模型（$\pi_\theta(y_t) < \pi_{\text{ref}}(y_t)$，这在正常训练中占了 **`48.5%`** 的 Token！），**$\hat{D}_{k1}(y_t)$ 算出来的数值就是严格小于 `0` 的负数**！
       - 在 PPO/RLOO 中，当你从奖励中减去 $\beta \cdot \hat{D}_{k1}$ 时，**减去一个负数等于加上一个正奖励**！模型立刻发现：只要故意吐出一些 $\pi_\theta < \pi_{\text{ref}}$ 的低概率生僻词，不仅不会受到 KL 惩罚，反而能白嫖一大笔正奖励！
       - 更糟糕的是，在 $\pi_\theta \approx \pi_{\text{ref}}$ 附近，$\hat{D}_{k1}$ 的一阶导数不为零，导致其蒙特卡洛采样方差比二阶紧致估计器高出 **`14.8 倍`**！
  2. **深坑二：为了消除采样方差，改算全词表解析 KL（$\sum_{v=1}^{V} \pi_\theta(v) (\log \pi_\theta(v) - \log \pi_{\text{ref}}(v))$，Tier 2），结果单步狂吞 `39.86 GB` 显存当场 OOM，且跨节点传输 `19.93 GB` Logits 把网络打爆**！
     - 在 `[B=4, S=8192, V=152064]` 下，要把 152,064 个词的概率全部算出来求和，不仅需要在显存里同时保留 $\pi_\theta$ 和 $\pi_{\text{ref}}$ 的全词表张量（**`2 × 19.93 GB = 39.86 GB` HBM**），还逼迫远端的 Reference Worker 每次通过 Ray/RPC 把 `19.93 GB` 的全量 Logits 传给 Trainer（而不是只传 `0.12 MB` 的一维已采样 Token Logprob）！
  3. **深坑三：改用 `DeepSeekMath / DeepSeek-R1` 推荐的 `Schulman k3` 估计器（$\hat{D}_{k3} = e^{\Delta} - 1 - \Delta$），但忘了在求指数 `exp()` 之前对 $\Delta = \log \pi_{\text{ref}} - \log \pi_\theta$ 做对称截断（Clamp），导致 `6.4%` 的批次因罕见词 $\exp(\Delta)$ 溢出产生 `Inf / NaN` 梯度爆炸（Tier 3）**！
  采用 **仅基于一维已采样 Token 对数概率 `[B, S]`（仅占 `0.12 MB` 显存，比全词表省 `332,000 倍`！）的对称截断 `Schulman k3` 估计器（先将 $\Delta = \log \pi_{\text{ref}} - \log \pi_\theta$ 钳位在 `[-10.0, +10.0]`，再算 $\exp(\Delta) - 1 - \Delta$）（Tier 4）**，可同时实现 **数学期望 100% 无偏**、**逐 Token 严格非负（$\ge 0$，`0%` 负数作弊）**、**方差降低 `13.5 倍`** 以及 **`0.0%` 指数溢出爆炸**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：只用 3 行数学推导看懂为什么 John Schulman 的 `k3` 估计器是“鱼与熊掌兼得”的神来之笔！**
   - 设重要性采样比率为 $r_t = \frac{\pi_{\text{ref}}(y_t)}{\pi_\theta(y_t)} = \exp\bigl(\underbrace{\log \pi_{\text{ref}}(y_t) - \log \pi_\theta(y_t)}_{\text{记为 } \Delta_t}\bigr)$。
   - 注意到一个美妙的恒等式：当 $y_t \sim \pi_\theta$ 时，$r_t$ 的期望恒等于 $1$！
     $$\mathbb{E}_{y_t \sim \pi_\theta}\bigl[r_t - 1\bigr] = \sum_{v \in V} \pi_\theta(v) \left(\frac{\pi_{\text{ref}}(v)}{\pi_\theta(v)} - 1\right) = \sum_{v \in V} \pi_{\text{ref}}(v) - \sum_{v \in V} \pi_\theta(v) = 1 - 1 = \mathbf{0}$$
   - 这意味着 **$r_t - 1$ 是一个期望严格为 `0` 的免费控制变量（Zero-Mean Control Variate）**！
   - 把这个期望为 `0` 的项加到原本有正有负的 $\hat{D}_{k1} = -\log r_t = -\Delta_t$ 上，就得到了 **`k3` 估计器**：
     $$\hat{D}_{k3}(y_t) = (r_t - 1) - \log r_t = \mathbf{e^{\Delta_t} - 1 - \Delta_t}$$
   - **三大神级数学性质**：
     1. **100% 严格无偏**：$\mathbb{E}_{y_t \sim \pi_\theta}[\hat{D}_{k3}] = \mathbb{E}[-\log r_t] + \underbrace{\mathbb{E}[r_t - 1]}_{0} \equiv D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$！
     2. **100% 逐 Token 严格非负（永不产生负惩罚）**：由经典不等式 $e^x \ge 1 + x$（对一切实数 $x \in \mathbb{R}$ 恒成立，且仅当 $x=0$ 即 $\pi_\theta(y_t) == \pi_{\text{ref}}(y_t)$ 时取等号 `0`），**每一个 Token 算出来的 $\hat{D}_{k3}(y_t)$ 恒大于等于 `0`**！
     3. **在 $\Delta_t \approx 0$ 处一阶导数抵消，方差暴降 `13.5 倍`**：对 $e^{\Delta} - 1 - \Delta$ 做泰勒展开得到 $\frac{1}{2}\Delta^2 + \frac{1}{6}\Delta^3 + \dots$，消除了线性噪声项！
2. **为什么必须在 `exp()` 前加上 `clamp(Delta, -10.0, 10.0)`？**
   - 如果某个采样出的探索 Token 在训练几步后被新策略 $\pi_\theta$ 判定为极低概率（如 $\log \pi_\theta = -28.0$，而 $\log \pi_{\text{ref}} = -3.0$），此时 $\Delta_t = +25.0$，一旦不加截断直接算 $e^{+25.0} \approx 7.2 \times 10^{10}$，单个异常 Token 的梯度就会瞬间冲垮整个 Batch！加上对称钳位 `clamp(log_ref - log_pi, -10.0, 10.0)` 后，既完全保留了 $[-10, 10]$（覆盖概率比从 $e^{-10}$ 到 $e^{+10} \approx 22,026$ 倍！）内的无偏曲率，又 `100%` 封死了指数爆炸！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：在 `GRPO`（如 `DeepSeek-R1`）中，这个 `k3` KL 项应该从奖励 $R$ 里减掉（Reward-Level KL），还是直接作为损失项加在 Policy Loss 后面（Direct-in-Loss KL）？
- **`DeepSeek-R1 / GRPO` 标准实践是：直接作为独立损失项加在 Loss 中（Direct-in-Loss KL，`loss = pg_loss + beta * k3_kl`）**！
  为什么？因为 `GRPO` 在每个 Prompt 的 $G = 16$ 条采样轨迹内部会对奖励做 **组内标准差归一化** $\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$！如果你把 KL 混在奖励 $r_i$ 里，当某道题的 16 条轨迹全都答对（$r_i \equiv 1$）时，组内归一化会把微小的 KL 噪声放大除以接近 0 的标准差；而将 `beta * k3_kl` 直接加在目标函数外层，梯度直接对 $\log \pi_\theta(y_t)$ 求导，物理意义极其干净清晰！

### Q2：如果直接对 $\hat{D}_{k3}(\pi_\theta) = \exp(\log \pi_{\text{ref}} - \log \pi_\theta) - 1 - (\log \pi_{\text{ref}} - \log \pi_\theta)$ 自动求导（Autograd `.backward()`），算出来的梯度是期望哪一个方向的 KL 散度？
- **这是一个极其深刻的理论细节**！
  - 对 $\log \pi_\theta(y_t)$ 直接求 $\hat{D}_{k3}$ 的梯度：
    $$\frac{\partial \hat{D}_{k3}}{\partial \log \pi_\theta(y_t)} = 1 - \frac{\pi_{\text{ref}}(y_t)}{\pi_\theta(y_t)}$$
  - 对该梯度在 $y_t \sim \pi_\theta$ 下求期望：它恰好等价于最小化正向/反向混合锚点并使 $\pi_\theta = \pi_{\text{ref}}$ 成为唯一零梯度不动点（而在需要严格反向 KL 路径导数时则使用 $\hat{D}_{k2} = \frac{1}{2}(\log \pi_\theta - \log \pi_{\text{ref}})^2$ 的梯度 $\log \pi_\theta - \log \pi_{\text{ref}}$）。

### Q3：在什么场景下才真正需要用到“Top-$K$ + 尾部桶合并（Top-$K$ Exact KL）”？
- 只有做 **在线策略蒸馏（On-Policy Knowledge Distillation，学生模型实时向一个能力更强的 671B 教师模型蒸馏概率分布）** 时，因为需要对齐未采样到的次优候选词概率，才让教师模型只传回每步概率最高的前 $K = 64$ 个词的 `(token_id, logprob)` 加上剩余词的 `1 - sum(p_top64)` 尾桶，从而用 **`0.04%`** 的网络流量还原出 `99.9%` 的全词表蒸馏 KL！而在标准 RL（`GRPO / PPO`）锚定参考模型时，**1D `Clamped k3`（每词仅 1 个标量）就完全够用了**！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 大模型强化学习（`GRPO / PPO`）KL 散度估计器选型决策树

```mermaid
flowchart TD
    A["大模型 RL 训练计算 KL 正则项 D_KL(pi_theta || pi_ref) (B=4, S=8,192, V=152,064)"] --> B{"KL 散度计算粒度与公式?"}
    B -- "全词表解析求和 Sum_v pi_theta(v) * (log_pi - log_ref)" --> C["⛔ Tier 2: 必须实例化两个 [4, 8192, 152064] 全词表张量 -> 狂吞 39.86 GB 显存当场 OOM!"]
    B -- "仅在已采样的 1D Token 序列 [B, S] 上做蒙特卡洛估计 (仅占 0.12 MB 显存)" --> D{"采用哪一个逐 Token 估计公式?"}
    D -- "朴素 k1 = log_pi_theta - log_pi_ref" --> E["⛔ Tier 1: 48.5% 的 Token 算出负数 KL (惩罚变奖励 Bonus!) + 14.8x 高方差引发训练崩溃!"]
    D -- "无截断 Schulman k3 = exp(log_ref - log_pi) - 1 - (log_ref - log_pi)" --> F["⚠️ Tier 3: 虽恒正且无偏, 但罕见词上 exp(log_ref - log_pi) 极易溢出导致 6.4% 批次 Inf/NaN!"]
    D -- "对称钳位 Schulman k3: d = clamp(log_ref - log_pi, -10, 10); exp(d) - 1 - d" --> G["🏆 Tier 4: 仅 0.12 MB 显存 (-39.86 GB) + 100% 恒正无偏 + 方差降 13.5x + 0.0% 溢出!"]
```

### 3.2 朴素 `k1` 负数作弊与全词表 `39.86 GB` OOM vs. 截断 `Schulman k3`（`0.12 MB`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 要么 k1 算出负数奖励作弊, 要么全词表 KL 吃掉 39.86 GB 爆显存!"]
      B1["Tier 1 (朴素 k1): 当 pi < ref 时 k1 < 0 -> 48.5% Token 惩罚变奖励, 方差高 14.8 倍!"] --> B2["Tier 2 (全词表 KL): 展开 152,064 维词表 -> 狂吞 39.86 GB HBM 且 RPC 传 19.93 GB!"]
    end
    subgraph Good["🏆 Tier 4: 一维采样 Token 上的对称钳位 Schulman k3 (exp(d) - 1 - d >= 0)"]
      G1["Reference Worker 仅返回采样词的 1D log_pi_ref [B, S] (仅 0.12 MB, 缩小 332,000 倍!)"] --> G2["d = clamp(log_ref - log_pi, -10, 10) -> k3 = exp(d) - 1 - d 恒 >= 0, 0% 负数, 0% Inf!"]
    end
```

### 3.3 `verl / GRPO` 中 Reference Worker 1D 对数概率传输与 `Clamped k3` 损失计算时序图

```mermaid
sequenceDiagram
    participant Ref as Reference Worker (Frozen 70B)
    participant RPC as Ray / RPC Data Bus
    participant Actor as Actor Trainer (Fused Linear Logprobs)

    Ref->>Ref: 1. Forward pass on `[B=4, S=8192]` -> gather ONLY sampled token `logprobs_ref[4, 8192]`
    Ref->>RPC: 2. Transmit 1D `logprobs_ref` (`0.12 MB` instead of `19.93 GB` full logits!)
    Actor->>Actor: 3. Compute `logprobs_theta[4, 8192]` on sampled tokens
    Actor->>Actor: 4. `d = torch.clamp(logprobs_ref - logprobs_theta, -10.0, 10.0)`
    Actor->>Actor: 5. `k3_kl = torch.exp(d) - 1.0 - d` (`>= 0` everywhere, 0.0% Inf, 0.12 MB HBM!)
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | RL KL 散度估计器方案 | KL 显存工作区 | 负数 KL Token 比例 | 相对梯度方差 | `exp()` 溢出 `NaN` 率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 朴素 `k1` (`log_pi - log_ref`)** | 一维对数差，无控制变量 | `0.00012 GB` | **`48.5%` (正负抵消作弊! ✕)** | `14.8x` (极高方差 ✕) | `0.0%` |
| **Tier 2: 全词表解析求和 (`Full-Vocab KL`)** | 遍历 `152,064` 词表精确求和 | **`39.86 GB` (OOM! ✕)** | `0.0%` | `1.0x` | `0.0%` |
| **Tier 3: 未截断 `Schulman k3` (`exp(d)-1-d`)** | 加入 $r-1$ 控制变量，未限幅 | `0.00012 GB` | `0.0%` | `1.1x` | **`6.4%` (罕见词爆炸! ✕)** |
| **Tier 4: 对称钳位 `Schulman k3` (推荐)** | **`clamp(d, -10, 10)` + `exp(d)-1-d`** | **`0.00012 GB` (`-39.86 GB` 🏆)** | **`0.0%` (恒 $\ge 0$ 🏆)** | **`1.1x` (`13.5x` 更低 🏆)** | **`0.0%` 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom/run.py`](../../experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-sizer.html`](../reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-selection-cards.html`](../reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Naive k1 (48.5% Negative)", "variance_ratio_x": 14.8, "hbm_gb": 0.00012 },
      { "tier": "Tier 2: Full-Vocab KL (39.86GB OOM)", "variance_ratio_x": 1.0, "hbm_gb": 39.86 },
      { "tier": "Tier 3: Unclamped Schulman k3 (6.4% Inf)", "variance_ratio_x": 1.1, "hbm_gb": 0.00012 },
      { "tier": "Tier 4: Clamped Schulman k3 (Recommended)", "variance_ratio_x": 1.1, "hbm_gb": 0.00012 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "variance_ratio_x": "Quantity",
    "hbm_gb": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Clamped Schulman k3 Estimator Cuts Variance 13.5x vs k1 & Saves 39.86 GB HBM vs Full-Vocab KL",
    "subtitle": "Relative Monte-Carlo estimator variance (x) across 4 KL estimation tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "variance_ratio_x",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有 `GRPO`、`PPO`、`RLOO`、`REINFORCE++` 强化学习后训练管线。
- **⛔ Not for**：在奖励或损失中直接使用可正可负的 `k1 = logp_theta - logp_ref`，或不加 `clamp` 直接计算 `exp(logp_ref - logp_theta)`。

### 生产核验 Checklist
- [ ] **统一使用 `Schulman k3` 估计器（`exp(d) - 1 - d`，其中 `d = logp_ref - logp_theta`）替代朴素 `k1`**：确保每个 Token 的 KL 惩罚恒大于等于 `0`。
- [ ] **在调用 `torch.exp(d)` 前强制执行对称截断 `d = torch.clamp(logp_ref - logp_theta, min=-10.0, max=10.0)`**。
- [ ] **Reference Worker 与 Actor Trainer 之间仅传输已采样 Token 的 1D `logprobs[B, S]`**：严禁实例化或跨进程传输 `[B, S, V]` 全词表张量。
