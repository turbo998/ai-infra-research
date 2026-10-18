# Zero-Bubble 流水线并行（`ZB-H1/ZB-H2/ZB-V`）与 DeepSeek-V3 `DualPipe`/`DualPipeV` 双向计算-通信完全重叠架构

- **关联问题 ID**：`zero-bubble-dualpipe-1f1b-interleaved-pp` (`priority: 17`)
- **关联实验**：[`experiments/zero_bubble_dualpipe/run.py`](../../experiments/zero_bubble_dualpipe/run.py) · [`results.json`](../../experiments/zero_bubble_dualpipe/results.json)
- **一手验证源码**：
  - `deepseek-ai/DualPipe` (`commit 030ce4325f4ebeb437da4ebc6d00a70469dd58ae`)：
    - `dualpipe/dualpipe.py` (`blob 686fb533905da5d8c1111d3a9e31d650ea7c1c0b`)
    - `dualpipe/dualpipev.py` (`blob cb8dc762c4fdc91cdb8bd1a52a6ef9067c256d66`)
    - `dualpipe/utils.py` (`blob 5b1a746c7203cbff0a710361ab0e8fb76c9df2cb`)
    - `README.md` (`blob d367fd37e728ca651b43b6b29a5fc55c3771d803`)
  - `sail-sg/zero-bubble-pipeline-parallelism` (`commit 39f2c186a27580b7051eb51e0f651e1efd0a8170`)：
    - `megatron/core/pipeline_parallel/zerobubble/scheduler/zb.py` (`blob b056cde1881870f6717232523a62b610117193db`)
    - `megatron/core/pipeline_parallel/zerobubble/scheduler/zbv.py` (`blob b76a8635f8da8818c0103a3b8636829603138d46`)
    - `megatron/core/optimizer/zbpp_optimizer_helper.py` (`blob d2fe861039da4bf9836d3b84c826b5ccce6bd1a8`)

---

## 1. 架构背景：为何标准 `1F1B` 与 Interleaved `1F1B` 在大规模 MoE 训练中遭遇双重瓶颈？

在千亿至万亿参数模型（如 DeepSeek-V3 `671B`）跨节点训练中，Pipeline Parallelism (`PP`) 面临两大结构性损耗：

1. **反向传播粗粒度绑定导致的流水线气泡（Pipeline Bubble）**：传统 `1F1B` 将反向传播视为不可分割的原子块 $B_{\text{full}}$，导致流水线首尾产生 $(PP - 1)(F + B_{\text{full}})$ 的固有空闲气泡。即使采用 Megatron Interleaved `1F1B`（虚拟流水线分片数 $v$），气泡虽降至 $\frac{1}{v}(PP - 1)(F + B_{\text{full}})$，但跨节点 P2P 通信次数暴增 $v$ 倍，且峰值激活显存膨胀至 $(1 + \frac{PP-1}{PP \cdot v}) \cdot PP$。
2. **MoE 跨节点 `EP All-to-All` 通信无法被单向流水线掩盖**：在大规模 MoE（如 `EP=64`）中，每个 Transformer 层包含 2 次 `All-to-All`（Dispatch 与 Combine），通信耗时占计算耗时的 `30%–50%`。在单向流水线中，同一时刻单个 GPU 只执行单个微批次的 `F` 或 `B`，同一微批次内部的 `Computation -> Dispatch -> Expert MLP -> Combine` 存在严格的串行数据依赖，无法实现 100% 重叠。

---

## 2. 一手源码解剖：`B/W` 算子解耦、双向 `DualPipe` 8 步状态机与 `Post-Validation Rollback`

### 2.1 Zero-Bubble：$B$ (`dgrad`) 与 $W$ (`wgrad`) 解耦及 `WeightGradStore` FIFO 队列

对于线性层 $Y = X W^\top$，反向传播实际上由两个独立算子组成：
- **激活梯度 $B$ (`dgrad`)**：$\nabla_X = \nabla_Y W$，**必须立即计算并发送给上游 PP 阶段**以驱动上游微批次反向传播；
- **权重梯度 $W$ (`wgrad`)**：$\nabla_W = \nabla_Y^\top X$，**不处于跨阶段关键路径上**，只要在本轮 Optimizer Step 更新该层参数前完成即可！

在 `deepseek-ai/DualPipe` 的 `dualpipe/utils.py`（第 `8–34` 行）与 `sail-sg/zero-bubble-pipeline-parallelism` 的 `zbpp_utils.py` 中，`WeightGradStore` 通过静态队列拦截线性层的 $\nabla_W$ 计算闭包：

```python
# dualpipe/utils.py (lines 8-29)
class WeightGradStore:
    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()
```

当 `_backward_compute_chunk(phase, enable_zb=True)` 执行时，算子仅计算 $\nabla_X$ 并将 $\nabla_W$ 函数压入 `WeightGradStore.funcs_queue`；随后在原本会产生流水线气泡的空闲窗口调用 `_weight_chunk() -> WeightGradStore.pop()` 异步消费积压的 $W$ 任务，并在计算完 $W$ 后立即释放保存的前向激活张量 $X$。

### 2.2 DeepSeek-V3 `DualPipe` 与 `DualPipeV` 源码 8 步调度状态机

在 `dualpipe/dualpipe.py`（第 `330–426` 行）中，`DualPipe` 在偶数个 PP Rank（`num_ranks % 2 == 0`）且微批次 `num_chunks >= num_ranks * 2` 时，从流水线两端（Rank `0` 和 Rank `PP-1`）同时对称注入微批次（Phase `0` 与 Phase `1`），并通过 8 个精确阶段推进：

1. **Step 1 (`nF0`)**：执行 `(num_half_ranks - half_rank - 1) * 2` 次单向 `F0` 预热。
2. **Step 2 (`nF0F1`)**：执行 `half_rank + 1` 次正反双向交错前向预热。
3. **Step 3 (`nB1W1F1`, `enable_zb=True`)**：执行 `num_half_ranks - half_rank - 1` 次零气泡填充（将 `B1` 拆出 `W1` 填补 `F1` 等待气泡）。
4. **Step 4 (`Main step: nF0B1F1B0`)**：稳态主循环（执行 `half_num_chunks - num_ranks + half_rank + 1` 轮），每轮调用 `_forward_backward_chunk(0, 1)` 与 `_forward_backward_chunk(1, 0)`，将方向 `0` 的前向块与方向 `1` 的反向块绑定在 `overlapped_forward_backward` 中执行——**当方向 0 处于 `EP All-to-All` 通信时，方向 1 正在执行 Attention/MLP 计算，反之亦然，从而将跨节点 `EP All-to-All` 与 P2P 通信 100% 掩盖在对方计算之下**！
5. **Step 5 (`nB1F1B0`) $\rightarrow$ Step 6 (`nB1B0`, 后半段开启 `enable_zb=True`) $\rightarrow$ Step 7 (`nWB0`, `enable_zb=True`) $\rightarrow$ Step 8 (`nW`)**：排空全部剩余反向块与 `WeightGradStore` 队列（第 `425` 行断言 `assert WeightGradStore.funcs_queue.empty()`）。

而在 `dualpipe/dualpipev.py`（第 `288–397` 行）与 `README.md`（第 `26–32` 行）中，**`DualPipeV`** 采用 Sea AI Lab 的 `"Cut-in-half"` V 型折叠变换，仅用 `PP/2` 台设备（每台设备承载正向阶段 $r$ 与反向阶段 $PP-1-r$ 两个非对称不同层分片）即可达到与 `DualPipe` 完全相同的气泡上限 $(\frac{PP}{2}-1)(F\&B + B - 3W)$，且**无需像 `DualPipe` 那样在集群两端复制两份模型参数（集群总参数副本从 `2x` 降回 `1x`）**！

| 调度算法（基于相同阶段数 $PP$） | 流水线计算气泡公式 | 单设备参数量 | 峰值激活块数 | 所需设备数 | 跨节点 EP 通信掩盖 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `1F1B` | $(PP-1)(F + B)$ | $1\times$ | $PP$ | $PP$ | 无（串行暴露） |
| `ZB-H1` (`ZB1P`) | $(PP-1)(F + B - 2W)$ | $1\times$ | $PP$ | $PP$ | 部分 P2P |
| `ZB-V` | $\approx 0$（$W$ 双向填谷） | $1\times$ | $PP$ | $PP/2$ | 仅掩盖 P2P |
| `DualPipe` | $(PP/2-1)(F\&B + B - 3W)$ | $2\times$ | $PP+1$ | $PP$ | **100% 双向 `F&B` 重叠** |
| `DualPipeV` | $(PP/2-1)(F\&B + B - 3W)$ | $2\times$（单卡含 2 阶段，集群总计 $1\times$） | $PP+1$ | $PP/2$ | **100% 双向 `F&B` 重叠** |

### 2.3 `Post-Validation Optimizer Rollback`：消除优化器更新前的全局同步气泡

在 Zero-Bubble 调度中，不同 Stage 完成最后一批 $W$ 的时刻相差 $(PP-1)W$。如果在 `optimizer.step()` 之前像传统 Megatron 那样执行全局梯度范数/NaN `AllReduce` 校验，所有提前算完的 Stage 必须空等最晚的 Stage，从而使刚刚消除的气泡重新出现！

`sail-sg/zero-bubble-pipeline-parallelism` 在 `megatron/core/optimizer/zbpp_optimizer_helper.py`（第 `6–147` 行）中实现了 **先乐观执行本阶段 `AdamW` 更新，同时异步发起全局 `NaN/Inf` 与 Gradient Norm 归约；若下一个 Step 开头发现上一步校验失败，则调用 `rollback_adamw` 精确逆运算回滚参数与动量状态**：

$$\begin{aligned}
u_t &= \text{lr} \cdot \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon} \\
p_{t-1} &= \frac{p_t + u_t}{1 - \text{lr} \cdot \lambda}, \quad
v_{t-1} = \frac{v_t + (\beta_2 - 1) g_t^2}{\beta_2}, \quad
m_{t-1} = \frac{m_t + (\beta_1 - 1) g_t}{\beta_1}
\end{aligned}$$

---

## 3. 离线实验验证结果（`experiments/zero_bubble_dualpipe`）

运行 `python experiments\zero_bubble_dualpipe\run.py --check`（配置 $PP = 16, M = 64, F = 10\text{ ms}, B_{\text{dgrad}} = 10\text{ ms}, W_{\text{wgrad}} = 9\text{ ms}, T_{\text{comm}} = 9.5\text{ ms}$）得到如下对比：

| 调度策略 (`PP=16, M=64`) | 计算气泡 (`ms`) | 暴露通信 (`ms`) | 总 Step 耗时 (`ms`) | 气泡率 (`%`) | 暴露通信占比 (`%`) | 峰值激活显存 (`GiB`) | 相对 `1F1B` 净加速 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1F1B` | `435.00` | `615.41` | `2,906.41` | `14.97%` | `21.17%` | `13.60 GiB` (`16` 块) | `1.000x` |
| `Interleaved-1F1B (v=2)` | `217.50` | `747.18` | `2,820.68` | `7.71%` | `26.49%` | `14.45 GiB` (`17` 块) | `1.030x` |
| `ZB-H1 (ZB1P)` | `165.00` | `360.24` | `2,381.24` | `6.93%` | `15.13%` | `13.60 GiB` (`16` 块) | `1.221x` |
| `ZB-H2` | `30.00` | `315.21` | `2,201.21` | `1.36%` | `14.32%` | `26.35 GiB` (`31` 块) | `1.320x` |
| `ZB-V` | `27.00` | `262.68` | `2,145.68` | **`1.26%`** | `12.24%` | `13.60 GiB` (`16` 块) | `1.355x` |
| `DualPipe` (`2x` 参数副本) | `88.20` | **`9.12`** | **`2,008.71`** | `4.39%` | **`0.45%`** | `14.45 GiB` (`17` 块) | **`1.447x`** |
| `DualPipeV` (`1x` 集群参数) | `88.20` | **`10.64`** | **`2,010.23`** | `4.39%` | **`0.53%`** | `14.45 GiB` (`17` 块) | **`1.446x`** |

- **`Post-Validation Rollback` 精度与吞吐验证**：在 FP64/FP32 数值验证中，`rollback_adamw` 逆变换的最大绝对误差仅为 `6.94e-18`；在 `PP=16`、每步异常概率 `0.15%` 下，异步后验回滚消除了 `149.0 ms/step` 的末尾 $W$ 排空等待，带来 **`+7.39%`** 的端到端训练吞吐提升。

---

## 4. 解决方案架构师行动指南与设计练习

1. **稠密模型 vs 大规模 MoE 模型的流水线选型**：
   - 若训练**稠密模型**或节点内 `EP <= 8`（无跨节点 `All-to-All` 瓶颈），优先选择 **`ZB-V`**（`1.26%` 气泡率、无参数复制、`PP` 块恒定激活显存）；
   - 若训练**跨节点大规模 MoE 模型**（`EP >= 32`，跨节点 `All-to-All` 占比高），必须采用 **`DualPipeV`**（当层数可被 $2 \times \text{Ranks}$ 整除时）或 **`DualPipe`**，通过双向 `F&B` 算子重叠将暴露通信压降至 `<0.6%`。
2. **可证伪后续问题**：当结合 Context Parallelism (`CP`) 处理 `128K+` 变长序列且各微批次有效序列长度不均匀时，双向 `DualPipe` 的 `F0` 与 `B1` 执行耗时失配会如何放大同步气泡？
