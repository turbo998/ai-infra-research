# 2D 混合序列并行：`USP = Ulysses × ZigZag Ring-Attention` 超长上下文训练架构

> **研究问题 ID**：`context-parallel-ring-ulysses-usp-long-seq`（`priority: 19`）
> **关联组件**：`long-context-attention`
> **一手源码核验**：
> - `feifeibear/long-context-attention` (`commit a5f8f600d435ced180a97c91c68e87ff3cfb8f89`)：
>   - `yunchang/hybrid/attn_layer.py` (`blob 127674aa69cb6088b4968352a7e6cfe4cd54fb00`)：`LongContextAttention.forward` 正交组合 `PROCESS_GROUP.ULYSSES_PG` 与 `PROCESS_GROUP.RING_PG`。
>   - `yunchang/ring/zigzag_ring_flash_attn.py` (`blob 7fde360895136df7b1384f9d67ee4040fa74f92e`)：`zigzag_ring_flash_attn_forward` 与 `zigzag_ring_flash_attn_backward` 基于 `block_seq_len = q.shape[1] // 2` 的三阶段因果对称切分。
>   - `yunchang/comm/extract_local.py` (`blob 40963c484aed6c5c9b807d18e494dd1a45fce7f7`)：`zigzag_extract_local` 对称子块抽取 `[value_chunks[r_rank], value_chunks[2 * rd - r_rank - 1]]`。
> - `pytorch/torchtitan` (`commit e61df55b600331defa37e6997a4fdfc91cfaec3b`)：
>   - `torchtitan/distributed/context_parallel.py` (`blob 0f1c734d63417afba4e11a1228299e06cb2b5c9a`)：`HeadTailCPLoadBalancer`（强制要求 `seq_len % (2 * cp_size) == 0`）。
> - `NVIDIA/Megatron-LM` (`commit ce852bfb93f364f64a556602fc71a5d9ed2c377e`)：
>   - `megatron/core/context_parallel/layout.py` (`blob d6680d04c657d1320e7a1427b555a7a3f64953d0`)：`CPLayout = Literal["contiguous", "zigzag"]`、`THDCPLayoutPlan` 与 `_local_segment_ids`。
> **可复现离线实验**：[`experiments/context_parallel_usp_zigzag/run.py`](../../experiments/context_parallel_usp_zigzag/run.py)（`--check` 封印结果：[`results.json`](../../experiments/context_parallel_usp_zigzag/results.json)）

---

## 1. 为什么单一维度的 Context Parallelism 在 `1M–4M` GQA 模型上会失效？

当序列长度 $S$ 扩展到 $1,048,576$ (`1M`) 乃至 $4,194,304$ (`4M`) tokens 时，单层自注意力的计算复杂度按 $\mathcal{O}(S^2 d)$ 二次方暴涨，而单卡激活与 KV 缓冲区占用按 $\mathcal{O}(S)$ 增长。工业界早期的两种 1D 序列/上下文并行方案在 `CP = 64`（`8 nodes × 8 GPUs/node`）规模下各自撞上物理硬墙：

### 1.1 纯 DeepSpeed-Ulysses 的 GQA 头数天花板（$U \le H_{\text{kv}}$）与跨机 All-to-All 坍塌
DeepSpeed-Ulysses 在 Attention 前执行 `All-to-All(scatter_idx=head, gather_idx=seq)`，将序列完整拼回但把注意力头切分到 $U$ 张 GPU 上独立运行 FlashAttention。然而现代大模型普遍采用 Grouped-Query Attention（如 Llama-3-70B / DeepSeek GQA：$H_q = 64, H_{\text{kv}} = 8, d = 128$）：
- **GQA 头数天花板**：一旦 $U > H_{\text{kv}} = 8$（例如 $U = 64$），每个 Ulysses 分组被迫对 `K/V` 头执行 **$r_{\text{rep}} = U / H_{\text{kv}} = 8\times$ 冗余复制**，不仅使 `K/V` 的 All-to-All 通信字节膨胀 $8$ 倍，而且 `All-to-All` 必须跨越节点间 `400GbE RoCEv2`（有效带宽约 `45 GB/s`，远低于节点内 NVLink 4.0 的 `380 GB/s`）并遭遇多跳拥塞。
- **同步阻塞**：`All-to-All` 位于 FlashAttention 算子正前/正后的关键路径上，无法与当前层的 Attention 计算重叠。

### 1.2 朴素连续分块 Ring-Attention (`naive_ring_64`) 的 `~50%` 因果掩码空泡
朴素 Ring-Attention 将序列按连续区间切分为 $R$ 段：$[\,0..\frac{S}{R},\; \frac{S}{R}..\frac{2S}{R},\; \dots,\; \frac{(R-1)S}{R}..S\,]$。在因果下三角掩码（`causal=True`）下：
- 在 $R$ 步环形传递 `K, V` 过程中，对于第 $r$ 号 Rank（持有 $Q_r$）和来自第 $k$ 号 Rank 的 $K_k, V_k$：
  - 当 $k > r$（未来块）：因因果掩码全为 $-\infty$，Rank $r$ **完全跳过计算（0 FLOPs）**；
  - 当 $k == r$（对角块）：仅计算下三角，工作量为 $\frac{1}{2} (\frac{S}{R})^2$；
  - 当 $k < r$（历史块）：计算完整方阵，工作量为 $(\frac{S}{R})^2$。
- **木桶同步空泡**：每一步 Ring P2P 交换都必须等待执行完整方阵 $(\frac{S}{R})^2$ 的最慢 Rank（如 Rank $R-1$），而全集群每步平均有效因果工作量仅为 $\frac{R+1}{2R} (\frac{S}{R})^2$。因此朴素连续 Ring 的算法级空转气泡率高达：
  $$\text{Bubble}_{\text{contiguous}}(R) = \frac{R - 1}{2R} \times 100\% \quad \xrightarrow{R=64} \quad \mathbf{49.22\%}$$
- 同时，`63` 跳细粒度跨机 RoCE P2P 与失衡算子相互拉扯，使 `S=1M` 单层正向+反向步耗时劣化至 **`2,057.24 ms`**（有效注意力利用率仅 **`34.70%`**）。

---

## 2. 一手源码拆解：2D `USP (Ulysses × ZigZag Ring)` 如何同时打穿通信与因果失衡？

### 2.1 `yunchang/hybrid/attn_layer.py`：2D 正交进程组映射
在 `feifeibear/long-context-attention` (`commit a5f8f60`, `yunchang/hybrid/attn_layer.py`, `blob 127674aa`, L24–106) 中，`LongContextAttention` 将全局 `CP = U × R = 64` 拆解为两个正交进程组：
1. **节点内 `ULYSSES_PG` ($U = 8 = H_{\text{kv}}$)**：完全收敛在单机 8 卡 NVLink 4.0 (`380 GB/s`) 域内，恰好满足 $U = H_{\text{kv}} = 8$（**`kv_replication_factor = 1`**，零 KV 复制浪费）。
   - 前向入口执行 `SeqAllToAll4D.apply(self.ulysses_pg, q/k/v, scatter_idx=2, gather_idx=1)`，将局部张量形状从 `(B, S / 64, H_q=64, d)` 转换为 `(B, S / 8, H_q / 8 = 8, d)`，同时每个 Ulysses Rank 仅持有 **`1` 个 KV Head**（`(B, S / 8, H_kv / 8 = 1, d)`）。
2. **节点间 `RING_PG` ($R = 8$)**：在 `8` 个节点之间针对已按头切分后的单头 `K, V` 执行 8 步环形 P2P。因为 `Ulysses=8` 已经把每张卡负责的 KV 头数缩减了 $8$ 倍，**跨节点 RoCE Ring P2P 需要传输的 KV 字节数直接下降 $8\times$**！
3. 前向出口再通过逆向 `SeqAllToAll4D.apply(self.ulysses_pg, context_layer, gather_idx=1, scatter_idx=2)` 恢复 `(B, S / 64, H_q=64, d)`。

### 2.2 `zigzag_extract_local` 与 `Megatron-LM` `layout.py`：对称双块 `[r, 2R - 1 - r]` 切分
为了彻底消灭 Ring 维度的因果掩码失衡，`yunchang/comm/extract_local.py` (`blob 40963c48`, L29–50)、`torchtitan/distributed/context_parallel.py` (`blob 0f1c734d`, `HeadTailCPLoadBalancer`) 与 `NVIDIA/Megatron-LM` (`megatron/core/context_parallel/layout.py`, `blob d6680d04`, L83–100 `_local_segment_ids`) 均采用了 **ZigZag（Head-Tail）对称双子块切分**：

将全局序列划分为 $2R$ 个等长原子子块（对于 $R=8$，共 $16$ 个子块 $c_0, c_1, \dots, c_{15}$，每块长度 $L_{\text{sub}} = \frac{S}{2R}$）。第 $r \in \{0, \dots, R-1\}$ 号 Ring Rank 持有首尾对称的一对子块：
$$\text{Chunks}(r) = \bigl(\, c_{r},\; c_{2R - 1 - r} \,\bigr)$$
对于 $R=8$，各 Rank 持有的子块对为：`[0, 15], [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9], [7, 8]`。

**数学严格平衡证明**：
第 $i$ 个子块所需的因果 Token-Pair 计算面积正比于 $(i + 0.5) L_{\text{sub}}^2$。在 ZigZag 配对下，任意 Rank $r$ 的总因果计算面积为：
$$W_{\text{ZigZag}}(r) = \bigl[(r + 0.5) + (2R - 1 - r + 0.5)\bigr] L_{\text{sub}}^2 = \mathbf{2R \cdot L_{\text{sub}}^2 = \text{const.}}$$
与 Rank 编号 $r$ **完全无关**！如本仓离线验证 `verify_zigzag_chunk_balance(ring_degree=8)` 所证：
- 连续切分 `contiguous` 的 `max / avg` 负载比为 **`1.8750`**（`min / max = 0.0667`，空转气泡 **`46.67%`**）；
- 对称切分 `zigzag` 的 `max / avg` 与 `min / max` 均为严格 **`1.0000`**（算法级因果空泡 **`0.00%`**）。

### 2.3 `zigzag_ring_flash_attn.py`：三分支无空泡环形算子调度
在 `yunchang/ring/zigzag_ring_flash_attn.py` (`blob 7fde3608`, L26–73) 中，每个 Ring 步 `step`（共 $R$ 步）将本地 `q, k, v` 视为前后两个半块 `block_seq_len = q.shape[1] // 2`（即 $c_r$ 与 $c_{2R-1-r}$），并按 `step` 与 `comm.rank` 的相对关系执行三种高度对称的 FlashAttention 调用：
1. **`step == 0`（本地自注意）**：调用 `_flash_attn_forward(q, k, v, causal=True)`，计算 $c_r$ 对 $c_r$ 的下三角、$c_{2R-1-r}$ 对 $c_r$ 的全矩阵、以及 $c_{2R-1-r}$ 对 $c_{2R-1-r}$ 的下三角，合计精确等于 $2 \times \frac{1}{2} L_{\text{sub}}^2 + L_{\text{sub}}^2 = 2 L_{\text{sub}}^2$；
2. **`1 <= step <= comm.rank`（接收到的远端块拥有更小的前半块索引 $k_0 < r$，但后半块索引 $k_1 > 2R - 1 - r$）**：
   本地完整 `q`（含 $c_r$ 与 $c_{2R-1-r}$）只需对远端前半块 `k[:, :block_seq_len], v[:, :block_seq_len]` 执行无掩码 `_flash_attn_forward(q, k0, v0, causal=False)`，计算面积为 $(2 L_{\text{sub}}) \times L_{\text{sub}} = 2 L_{\text{sub}}^2$；
3. **`step > comm.rank`（接收到的远端块前后两半索引均小于 $2R - 1 - r$，但均大于 $r$）**：
   本地后半块 `q1 = q[:, block_seq_len:]`（即 $c_{2R-1-r}$）对远端完整 `k, v`（两半块）执行 `_flash_attn_forward(q1, k, v, causal=False)`，计算面积同样为 $L_{\text{sub}} \times (2 L_{\text{sub}}) = 2 L_{\text{sub}}^2$！

由此，**在全部 $R$ 个 Ring Step 的每一步中，所有 $R$ 张 GPU 执行的 FlashAttention 算子 FLOPs 完全恒定为 $2 L_{\text{sub}}^2$**，并通过 `update_out_and_lse(out, lse, block_out, block_lse)` 在线合并 Softmax Log-Sum-Exp。

---

## 3. 定量实验结果：`CP = 64` 下 `256K – 4M` 序列扩展矩阵

运行 `python experiments/context_parallel_usp_zigzag/run.py --check`（硬件基线：`8 nodes × 8 H100 SXM5`，节点内 NVLink `380 GB/s`，节点间 RoCEv2 `45 GB/s`，GQA `H_q=64, H_kv=8, d=128`），得到单层（Fwd + Bwd）端到端性能对比：

| 序列长度 $S$ | 并行策略 (`CP=64`) | $(U, R)$ 拓扑 | KV 复制倍数 | 因果失衡率 | 有效计算耗时 | 总通信耗时 | 暴露通信耗时 | 单层步耗时 | 注意力有效效率 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1,048,576` (`1M`)** | `ulysses_64_gqa_replicated` | `64 × 1` | `8x` | `0.00%` | `713.89 ms` | `75.77 ms` | `69.71 ms` | `783.59 ms` | `91.10%` |
| **`1,048,576` (`1M`)** | `naive_ring_64` | `1 × 64` | `1x` | `49.22%` | `1,405.81 ms` | `281.86 ms` | `651.43 ms` | `2,057.24 ms` | `34.70%` |
| **`1,048,576` (`1M`)** | `usp_contiguous_8x8` | `8 × 8` | `1x` | `43.75%` | `1,269.13 ms` | `34.10 ms` | `2.78 ms` | `1,271.91 ms` | `56.13%` |
| **`1,048,576` (`1M`)** | **`usp_zigzag_8x8` (Ours)** | **`8 × 8`** | **`1x`** | **`0.16%`** | **`715.03 ms`** | **`34.10 ms`** | **`2.36 ms`** | **`717.39 ms`** | **`99.51%`** |
| **`4,194,304` (`4M`)** | `ulysses_64_gqa_replicated` | `64 × 1` | `8x` | `0.00%` | `11,422.17 ms` | `303.07 ms` | `278.83 ms` | `11,701.00 ms` | `97.62%` |
| **`4,194,304` (`4M`)** | `naive_ring_64` | `1 × 64` | `1x` | `49.22%` | `22,492.89 ms` | `1,127.43 ms` | `9,827.36 ms` | `32,320.25 ms` | `35.34%` |
| **`4,194,304` (`4M`)** | `usp_contiguous_8x8` | `8 × 8` | `1x` | `43.75%` | `20,306.09 ms` | `136.40 ms` | `11.13 ms` | `20,317.21 ms` | `56.22%` |
| **`4,194,304` (`4M`)** | **`usp_zigzag_8x8` (Ours)** | **`8 × 8`** | **`1x`** | **`0.16%`** | **`11,440.45 ms`** | **`136.40 ms`** | **`9.46 ms`** | **`11,449.91 ms`** | **`99.76%`** |

### 核心架构结论
1. **`2.87x` (`1M`) 至 `2.82x` (`4M`) 端到端吞吐加速**：在 `S=1M` 下，`usp_zigzag_8x8` 将单层注意力步耗时从朴素 `naive_ring_64` 的 `2,057.24 ms` 压缩至 **`717.39 ms`**（提速 **`2.87x`**），在 `S=4M` 下从 `32,320.25 ms` 压缩至 **`11,449.91 ms`**（提速 **`2.82x`**）。
2. **跨节点 RoCE 流量锐减 `8.27x` 并完全隐藏**：由于节点内 `Ulysses=8` 先行按头切分，`RING_PG` 在节点间传递的 KV 字节数缩减至原来的 $1/8$；在 `S=1M` 下总通信耗时从 `281.86 ms` 降至 **`34.10 ms`**，且因每步 FlashAttention 负载严格均衡（残余失衡仅 **`0.16%`**），暴露通信仅剩 **`2.36 ms`**（注意力效率高达 **`99.51%`**）。
3. **Packed Variable-Length Sequence (`THD`) 布局转换代价**：如 `NVIDIA/Megatron-LM` `megatron/core/context_parallel/layout.py` (`THDCPLayoutPlan`) 所示，当一个 Batch 内打包了多条不等长文档（`cu_seqlens`）时，全局静态 `[r, 2R-1-r]` 切分会导致某条长文档全部落在同一对子块内。生产工程必须在 **每个单独文档内部** 执行 `2 * cp_size` 对称 ZigZag 切分（`THDCPLayoutPlan` 只在每个 Packed Batch 入口构建一次重排索引，并在各 Attention 层复用）。
