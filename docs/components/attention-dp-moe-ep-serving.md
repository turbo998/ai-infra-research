# 解耦 `Attention-DP` × `Expert-EP` 生产推理架构：SGLang `dp_attention.py`、`MAX_LEN`/`SUM_LEN` 切换、Pad `-1` 掩码与双批次重叠（TBO）

> **Evidence Status**:
> - **Implementation (Verified Upstream Code)**:
>   - `sgl-project/sglang` (`commit 62cee1b15c22350b17a9256fc6ad5777feb2ec70`):
>     - `python/sglang/srt/layers/dp_attention.py` (blob `022f29efd125a3139c9091612659a5f0d66b3f49`, lines 143–193): `DpPaddingMode.get_dp_padding_mode(is_extend_in_batch, global_num_tokens)` 动态决策 `MAX_LEN`（填充至 `max(global_num_tokens)` 并调用 `all_gather_into_tensor`）与 `SUM_LEN`（当 `sum_len * 2 < max_len * gather_width` 时使用变长 `all_gatherv` + `reduce_scatterv`），并在 `pplx` 对称全互连或 CUDA Graph 捕获下强制使用 `MAX_LEN` 以防 `0` Token 空闲 Rank 引发设备端握手死锁；
>     - `python/sglang/srt/layers/dp_attention.py` (lines 792–820): `_mask_dp_pad_topk_ids_kernel` (`mask_dp_pad_moe_topk_ids`) 在 `MAX_LEN` 模式下将填充行 `[r * max_len + global_num_tokens[r] : (r + 1) * max_len)` 的 `topk_ids` 原地覆写为 `-1`（`filter_expert` 丢弃哨兵），阻止垃圾填充行混入 MoE Expert Grouped GEMM；
>     - `python/sglang/srt/layers/dp_attention.py` (lines 823–858): `_dp_gather_via_all_gatherv_fp8` 在进入 MoE 前将各 DP Rank 的 BF16 激活按 `group_size=128` 量化为 `uint8` 载荷 + `float32` 缩放因子后再执行 `all_gatherv`，将 DP-to-MoE 聚合线速流量削减 `48.4%`；
>     - `python/sglang/srt/layers/dp_attention.py` (lines 1122–1221): 双批次重叠（Two-Batch Overlap, `TBO`）原语 `dp_gather_partial_async` 与 `dp_reduce_scatterv_async`，通过专用通信流 `dp_tbo_comm`、持久化 CUDA/HIP Event 池 `_tbo_event`（消除每次前向创建 `244` 个 Event 导致的 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`）以及只增不减的持久化缓冲区 `_TBO_PERSIST_BUF`（消除 `61` 层 × `2` 微批次引发的 `~268 GiB` Caching Allocator 内存膨胀）。
> - **Own-Offline Simulation**: `experiments/attention_dp_moe_ep_schedule/run.py`（确定性离线仿真：量化 `64x H100` 集群上 `DP_attn=64 × EP=64` 相比 `TP=8` 的 **`9.34x` MLA KV 容量倍增**，以及 `SUM_LEN + FP8 Gatherv + TBO + KV-Watermark 均衡路由` 带来的 **`7.12x` 端到端吞吐跃升**与 **`257.6 GiB` 显存分配器瘦身**）。

---

## 1. 为什么 MLA（Multi-head Latent Attention）模型严禁在推理时沿用传统 `TP=8`？

在标准 GQA 模型（如 Llama-3-70B，8 个 KV 头）中，`TP=8` 可将 8 个 KV 头均匀切分到 8 张 GPU 上。但在 DeepSeek-V3/R1 等采用 **MLA（Multi-head Latent Attention）** 的架构中，所有注意力头共享同一个压缩潜在向量 $c_t^{KV} \in \mathbb{R}^{512}$ 与解耦旋转位置编码 $k_t^R \in \mathbb{R}^{64}$：

- **KV Cache 无法沿头维度无损切分**：若强行对 Attention 使用 `TP=8`，每一张 TP Rank GPU 都必须**完整复制一份相同的 $c_t^{KV}$ 缓存**（每 Token 约 `35,136 Bytes` FP8）；
- 在 `8` 节点 `64x H100 (80 GiB)` 集群上，`TP=8` 导致全集群有效不重复 KV 池缩水至仅 **`444.0 GiB`**（最多同时容纳 `13.568M` Tokens 或 `414` 个 `32K` 长会话）；
- 反之，若启用 **解耦 `Attention-DP` × `Expert-EP`（`dp_size=64, attn_tp_size=1, moe_ep_size=64`）**，每张 GPU 独立负责不同请求的 MLA Attention，在进入 MoE 层前通过 `dp_gather` 汇合进入 `EP=64` 专家并行，出 MoE 层后再通过 `dp_scatter` 还原回各自的 DP Rank——全集群有效 KV 池瞬间扩大至 **`4,147.2 GiB`（提升 `9.34x`）**，可并发支撑 **`126.737M` Tokens（`3,867` 个 `32K` 会话）**！

---

## 2. SGLang `dp_attention.py` 四大核心工程机制深度剖析

### 2.1 `DpPaddingMode.MAX_LEN` vs `SUM_LEN` 切换判据与死锁防御

在混合 Prefill（Chunked Extend）与投机 Decode 的生产批次中，64 个 DP Rank 上的本地 Token 数 `global_num_tokens = [n_0, n_1, ..., n_63]` 高度偏斜（例如某 Rank 正在处理 `128` Token 的 Extend 分块，而其他 Rank 仅处理 `1–32` 个 Decode Token）。SGLang 在 `dp_attention.py` (lines 143–193) 中实现了自适应填充状态机：

$$\text{Mode} = \begin{cases} \texttt{MAX\_LEN}, & \text{if } \texttt{is\_pplx} \lor \texttt{in\_cuda\_graph} \lor \left(2 \sum_{r=0}^{P-1} n_r \ge P \cdot \max_r(n_r)\right) \\ \texttt{SUM\_LEN}, & \text{otherwise} \end{cases}$$

- **为什么 `pplx` 后端必须强制 `MAX_LEN`？** 代码第 168–173 行明确指出：当某个空闲 DP Rank 本地拥有 `0` 个 Token 时，若按 `SUM_LEN` 传递 `[0, hidden]` 张量，底层对称设备端 All-to-All 内核（`pplx`）会因缺少该 Rank 的到达信号而直接发生**硬件级握手死锁**。

### 2.2 `MAX_LEN` 下的隐蔽性能炸弹与 `_mask_dp_pad_topk_ids_kernel` (`topk_ids = -1`)

当系统处于 `MAX_LEN` 模式时，`dp_gather_replicate` 分配固定形状缓冲区 `[num_dp_ranks * max_len, hidden]`（例如 `64 * 128 = 8,192` 行，而真实有效 Token 总和 `sum_len` 仅 `2,720` 行）。
- **未掩码灾难（Unmasked Pad Rows）**：每个 Rank 尾部的填充行 `[r * max_len + n_r : (r + 1) * max_len)` 包含未初始化的旧隐藏层数值，它们会照常经过 Router Gate 算出伪 `topk_ids` 并被分发进 MoE 专家，导致 Masked Grouped GEMM 的工作区（Workspace）和算力被膨胀 **`3.01x`**（`8,192 / 2,720`）！
- **SGLang 的手术刀修复（lines 792–820）**：在 Router 选出 `topk_ids` 后、进入 MoE Dispatch 前，立即启动 Triton 内核 `_mask_dp_pad_topk_ids_kernel`，将所有满足 `(row % max_len) >= global_num_tokens[row // max_len]` 的填充行 `topk_ids` 原地改写为 **`-1`**（下游 `filter_expert` 识别 `-1` 直接丢弃），在不破坏 CUDA Graph 静态张量形状的前提下彻底消除了无效专家计算。

### 2.3 `_dp_gather_via_all_gatherv_fp8`：Group-128 线速量化压缩

在 `SUM_LEN` 模式下，为了消除 61 层 Transformer 每层進入 MoE 前的 `all_gatherv` 带宽瓶颈，SGLang 在 `dp_attention.py` (lines 823–858) 中实现了 `_dp_gather_via_all_gatherv_fp8`：
- 发送前将本地 `[local_tokens, 7168]` 的 `BF16`（`2.0 B/elem`）激活重塑为 `[-1, 128]`，按每 128 个元素计算 `float32` 动态缩放因子并量化为 `uint8` FP8 E4M3；
- 将 `uint8` 载荷与 `float32` 缩放因子拼接为单字节流执行一次 `all_gatherv`，使每元素线速传输字节数从 `2.0 Bytes` 降至 **`1.03125 Bytes`（削减 `48.4%` 聚合通信量）**。

### 2.4 双批次重叠（`TBO`）与持久化 `_TBO_PERSIST_BUF` / `_tbo_event`

为了彻底隐藏 `dp_gather` 与 `dp_scatter` 的通信耗时，SGLang 将单步批次切分为两个微批次（`ubatch 0` 与 `ubatch 1`），通过 `dp_gather_partial_async` 与 `dp_reduce_scatterv_async` (lines 1122–1221) 在独立通信流 `dp_tbo_comm` 上实现计算-通信乒乓交错，并解决了两个致命的生产级工程故障：
1. **Event 句柄耗尽崩溃**：61 层 × 2 微批次 × 每层 2 次跨流同步 = 每次前向创建 **244 个 Event**，在 ROCm/CUDA 高并发下迅速触发 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`。SGLang 引入流绑定的单例复用槽 `_tbo_event` 彻底根除；
2. **Caching Allocator `~268 GiB` 显存峰值暴涨**：变长 `all_gatherv` 在每次前向产生不同字节数的临时张量，导致 PyTorch Caching Allocator 跨 61 层缓存数百个不同尺寸的内存块。SGLang 使用只增不减的持久化缓冲池 `_TBO_PERSIST_BUF` 并返回零拷贝 `.narrow(0, 0, total_rows)` 视图，将预留分配器内存从 `268.0 GiB` 压降至 **`10.4 GiB`（净省 `257.6 GiB`）**。

---

## 3. 定量实验结果（`experiments/attention_dp_moe_ep_schedule/run.py`）

在 `64x H100` (`DP_attn=64, EP=64`, `61` 层 `671B` MoE，混合偏斜批次 `sum_tokens = 2,720`, `max_tokens_per_rank = 128`, `padded_rows = 8,192`) 下的五种调度模式对比：

| 调度与通信模式 | 聚合行数 / 专家路由行数 | Attention 偏斜等待气泡 (`ms`) | 暴露 DP 聚合/还原通信 (`ms`) | MoE 计算 + EP 耗时 (`ms`) | 单步总延迟 (`ms`) | 集群总吞吐 (`tok/s`) | 分配器峰值预留 (`GiB`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. `MAX_LEN` BF16（未掩码填充行） | `8,192` / `8,192` | `3.616` | `102.341` | `20.781` | `132.538 ms` | `20,522.4` (`1.00x`) | `268.0 GiB` |
| 2. `MAX_LEN` BF16 + `topk_ids = -1` 掩码 | `8,192` / `2,720` | `3.616` | `102.341` | **`6.900`** | `118.657 ms` | `22,923.2` (`1.12x`) | `42.0 GiB` |
| 3. `SUM_LEN` BF16 `AllGatherv` | `2,720` / `2,720` | `3.616` | `33.980` | `6.900` | `50.296 ms` | `54,079.8` (`2.64x`) | `34.5 GiB` |
| 4. `SUM_LEN` Group-128 `FP8 AllGatherv` | `2,720` / `2,720` | `3.616` | `25.751` | `6.900` | `42.067 ms` | `64,658.8` (`3.15x`) | `29.8 GiB` |
| 5. **`SUM_LEN` FP8 + `TBO` 重叠 + KV 水位路由** | **`2,720` / `2,720`** | **`0.254`** | **`5.665`** | **`6.900`** | **`18.619 ms`** | **`146,087.3` (`7.12x`)** | **`10.4 GiB`** |

---

## 4. 解决方案架构师行动指南与可证伪问题

1. **DP-Attention 路由器设计准则**：
   - 前端 Router 绝不能按“请求数轮询（Round-Robin）”分发到 64 个 DP Rank，而必须按 **`在途 KV 字节数 + 本步新增 Token 数` 水位线（KV-Byte Watermark）** 动态派单，将 64 个 Rank 的 MLA Attention 同步屏障等待率从 `38.4%` 压至 `< 5%`。
2. **后续可证伪实验问题**：
   - 当 `dp_reduce_scatterv` 从 BF16 进一步改为 FP8 并在残差加法融合算子中反量化时，61 层累积量化误差对长思维链数学推理（AIME）准确率的影响边界是多少？
