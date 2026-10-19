# 异步强化学习 (`Async RL`) 训推解耦架构：有界陈旧度 $\tau_{\text{max}}$、部分轨迹续写 (`Partial Rollout`) 与 Token 级重要性采样帕累托前沿

> **证据等级声明**：本文对 `AsyncRolloutWorker` 后台并发解耦、完成队列背压门控、同步/部分续写/全异步四象限调度及跨版本混合策略分布 $q(y\mid x)$ 的分析基于 `THUDM/slime`（commit `a4202e0595467d9374d9a68d45a34f45c3f8d230`）源码文件 `slime/rollout/fully_async_rollout.py`（blob `7a4ed8d13d6df8eb87cb557898a62466a29c3b89`）与 `docs/en/advanced/rollout-scheduling.md`（blob `a6d09fcec2f29cd8c8aee5528d70badc8c7f4625`）的一手核验（`implementation`）。长尾推理气泡率、陈旧度 $\tau_{\text{max}} \in \{0,1,2,4,8\}$ 下的 KL 散度与有效样本收敛速率来自确定性仿真算例 `experiments/async_rl_staleness_pareto/run.py`（`own-offline`）。

---

## 1. 为什么推理大模型（Reasoning RL / Agentic RL）在同步共置模式下遭遇 `76.8%` 的算力空转气泡？

在 DeepSeek-R1、GLM-4.5 及长程 Agentic 强化学习（GRPO / PPO）后训练中，单个训练步需要对每个 Prompt 采样 $G=8\text{–}16$ 条思维链轨迹（Rollout Group）。与预训练固定序列长度不同，推理轨迹长度服从极度偏斜的重尾帕累托分布（多数轨迹在 `2K–6K` tokens 结束，而少数困难数学/代码轨迹延伸至 `32K–64K` tokens）：
- 在 `experiments/async_rl_staleness_pareto/run.py` 建模的 64 组 GRPO 采样中，组平均完成时间为 **`13.70 s`**，P80 为 `17.50 s`，但最慢的长尾 Straggler 组耗时高达 **`59.15 s`**！
- 在传统的 **同步批次屏障基线（Synchronous Baseline, $\tau_{\text{max}} = 0$）** 下，所有 64 张 SGLang/vLLM 推理 GPU 必须等待最后一条 `59.15 s` 的轨迹跑完，才能触发训练更新（`14.0 s`）与 NCCL 权重广播（`1.35 s`）。这导致 Rollout 阶段的 **长尾等待气泡高达 `76.8%`**，推理 GPU 实际有效利用率跌至 **`23.2%`**（端到端单步耗时 `74.50 s`）。

---

## 2. 一手源码剖析：`THUDM/slime` 的两级解耦机制

### 2.1 `Partial Rollout`（部分轨迹跨步续写）与混合版本重要性采样

在 `THUDM/slime` 的 `docs/en/advanced/rollout-scheduling.md`（commit `a4202e059546`）中，定义了消除同步长尾气泡的第一种原语——**过采样 + 部分轨迹保存续写（`--partial-rollout` + `--over-sampling-batch-size`）**：
1. **机制**：当目标批次为 `64` 组时，系统同时发起 `80` 组（`1.25x` 过采样）并发生成。一旦前 `64` 组（对应 P80 分位耗时 `17.50 s`）生成并完成奖励打分，调度器立即向 SGLang 引擎发送 `abort_servers_until_idle`，将未完成的 16 组长尾轨迹的前缀 Token（长度 $k$）及其采样器元数据存入缓冲区，立即启动模型训练并将权重从版本 $v_0$ 更新至 $v_1$。
2. **跨版本混合行为分布 $q(y\mid x)$**：当未完成的轨迹在下一轮从第 $k+1$ 个 Token 处恢复生成时，整条轨迹的前半段由旧策略 $v_0$ 生成，后半段由新策略 $v_1$ 生成，其真实行为策略分布为：

$$q(y\mid x) = \prod_{t=1}^{k} q_{v_0}(y_t \mid x, y_{<t}) \prod_{t=k+1}^{T} q_{v_1}(y_t \mid x, y_{<t})$$

源码文档明确指出两种策略修正路径：
- **路径 A：Token 级对齐截断重要性采样（Token-Aligned TIS）**：每个采样 Token 严格保留其生成时刻对应策略版本的对数概率 $\log q_{v(t)}(y_t \mid h_t)$，训练时按逐 Token 重要性比率 $r_t(\theta) = \frac{p_\theta(y_t \mid h_t)}{q_{v(t)}(y_t \mid h_t)}$ 进行 PPO/GRPO 双重截断（Dual-Clip），**绝不能用单一当前采样器替代整条混合版本轨迹的分母**。
- **路径 B：旧前缀 Loss 掩码（`--mask-offpolicy-in-partial-rollout`）**：对 $t \le k$ 的旧版本前缀设置损失掩码 $m_t = 0$，仅对 $t > k$ 的新版本续写 Token 计算梯度 $\sum_t m_t \ell_t$。虽然避免了旧前缀的 Off-Policy 梯度偏差，但代价是**白白丢弃了前 $k$ 个已生成 Token 的监督学习信号**。

### 2.2 `AsyncRolloutWorker`：后台常驻池与双门控背压

在 `slime/rollout/fully_async_rollout.py`（blob `7a4ed8d13d6df8eb87cb557898a62466a29c3b89`）中，`AsyncRolloutWorker` 彻底解耦了 `max_concurrent_tasks` 与 `rollout_batch_size`：

```python
# Source: THUDM/slime slime/rollout/fully_async_rollout.py
while (
    len(active_tasks) < max_concurrent
    and self.output_queue.qsize() < max_concurrent
    and self.running
    and not self.paused
):
    groups = self.data_buffer.get_samples(1)
    ...
```

- **非阻塞完成回调 + 入队门控背压**：`self.output_queue` 故意设计为无界队列，避免在 asyncio 事件循环的 `add_done_callback` 中发生阻塞死锁；而真正的背压（Backpressure）设置在 `_loop()` 的拉取门控 `self.output_queue.qsize() < max_concurrent` 上——一旦已完成待消费的组数达到上限 $K = \lfloor C \cdot E / G \rfloor \times \text{prefetch\_factor}$，Worker 自动停止拉取新 Prompt，从而在系统层面严格界定了最大策略陈旧度 $\tau_{\text{max}} = v_{\text{train}} - v_{\text{oldest\_token}}$！
- **权重更新期间的平滑准入暂停（Admission Pause）**：在 NCCL 广播新权重前，控制器调用 `worker.pause(drain=False)` 暂停新任务准入，而被打断返回 `Sample.Status.ABORTED` 或 `PENDING` 的在途任务通过 `prepare_retry_group(retry_group, result)` 自动重入 `data_buffer`（保留同组内已完成的兄弟轨迹结果），实现零废算力热切换。

---

## 3. 陈旧度 $\tau_{\text{max}}$ 帕累托前沿：为什么 $\tau_{\text{max}} = 4$ 击败无界异步 $\tau_{\text{max}} = 8$？

我们在 `experiments/async_rl_staleness_pareto/run.py` 中对 $\tau_{\text{max}} \in \{0, 1, 2, 4, 8\}$ 进行了系统吞吐量与策略收敛效率的联合仿真：

| 调度模式 / 最大陈旧度上界 $\tau_{\text{max}}$ | 平均陈旧度 $\bar{\tau}$ (Versions) | 单步耗时 (`s`) | Rollout GPU 利用率 | 原始吞吐 (`steps/hr`) | 原始吞吐加速比 | 策略偏移 $D_{\text{KL}}(q \parallel p_\theta)$ | Dual-Clip 裁剪丢弃率 | 前缀掩码丢弃 Token 率 | **Token-Aligned TIS 有效收敛速率 (`eff_steps/hr`)** | **有效收敛加速比** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Sync Baseline ($\tau_{\text{max}}=0$)** | `0.00` | `74.50` | `23.2%` | `48.32` | `1.00x` | `0.00000` | `0.00%` | `0.00%` | `48.32` | **1.00x** |
| **Async ($\tau_{\text{max}}=1$)** | `0.58` | `40.60` | `33.1%` | `88.66` | `1.83x` | `0.00146` | `0.60%` | `8.41%` | `87.80` | **1.82x** |
| **Async ($\tau_{\text{max}}=2$)** | `1.16` | `29.36` | `46.5%` | `122.61` | `2.54x` | `0.00493` | `1.76%` | `16.82%` | `118.83` | **2.46x** |
| **Async ($\tau_{\text{max}}=4$, Pareto 最优)** | `2.32` | `23.12` | `59.8%` | `155.69` | `3.22x` | `0.01657` | `5.16%` | `33.64%` | **`140.73`** | **2.91x** |
| **Async ($\tau_{\text{max}}=8$, 过度陈旧退化)** | `4.64` | `19.66` | `71.1%` | `183.09` | `3.79x` | `0.05574` | `15.11%` | `55.00%` | `131.74` (`Masked: 83.41`) | **2.73x (退化)** |

### 核心架构结论
1. **原始吞吐 $\neq$ 有效策略收敛速度**：当放开队列背压至 $\tau_{\text{max}} = 8$（平均陈旧度 $\bar{\tau} = 4.64$ 步）时，虽然原始采样步速达到 `3.79x`（`183.09 steps/hr`），但策略偏移 $D_{\text{KL}}$ 飙升 `3.36x` 至 **`0.05574`**，触发 `15.11%` 的 Dual-Clip 截断丢弃，导致有效样本效率（ESS）跌至 `0.7195`，实际收敛速度反而比 $\tau_{\text{max}} = 4$ **倒退 6.4%**（从 `140.73` 跌至 `131.74 eff_steps/hr`）。
2. **Token-Aligned TIS 显著优于粗粒度前缀掩码**：在帕累托最优点 $\tau_{\text{max}} = 4$ 下，`--mask-offpolicy-in-partial-rollout` 因掩掉 `33.64%` 的旧版本前缀 Token，有效收敛速率仅为 `106.13 eff_steps/hr`；而保留逐 Token 生成版本 log-prob 的 **Token-Aligned TIS** 达到 **`140.73 eff_steps/hr`（提升 +32.6%，总有效加速比达 2.91x）**。
