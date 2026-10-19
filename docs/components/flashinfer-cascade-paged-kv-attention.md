# FlashInfer JIT Multi-Level Cascade Attention 与 Base-2 LSE 在线状态归约架构

> **证据分类**：`implementation`（基于 `flashinfer-ai/flashinfer` `commit f64b4ba8762ded12a4f4b9a2092974c9a395e7ca` 源码核验）+ `own-offline`（`experiments/flashinfer_cascade_paged_kv/run.py` 确定性数值与带宽仿真）。本文不宣称已完成生产集群真实 GPU 硬件性能剖析（`own-hardware`）。

---

## 1. 为什么 Paged KV 物理页去重不等于 HBM 带宽去重？

在多轮 Agent、RAG 检索增强或 CoT 树搜索（Tree-of-Thought / Best-of-$N$）推理场景中，同一批次（Batch Size $B \in [16, 256]$）的请求往往共享长达 $L_{\text{sys}} = 16,384$ tokens 的系统提示词（System Prompt）以及 $G$ 组长达 $L_{\text{rag}} = 8,192$ tokens 的文档上下文，而每个请求独有的对话后缀仅 $L_{\text{uniq}} = 1,024$ tokens。

虽然 vLLM / SGLang 的 Prefix Caching（RadixAttention）在物理显存中仅保存一份共享前缀的 KV 页（消除 HBM **容量**冗余），但在使用传统 Flat Paged-KV Decode Kernel 时：
1. 每个请求的 CTA（Thread Block）独立遍历其页表 `kv_indices`；
2. 由于 `80` 层 Llama-3.1-70B（`TP=4`，单卡每层每 Token KV 为 $2 \times 2 \times 128 \times 2 = 1,024\text{ B}$）的 $25,600$ tokens KV Cache 总量远超 H100 `50 MB` L2 Cache 容量，不同 CTA 无法跨层驻留复用共享前缀页；
3. 更关键的是，Flat Paged-KV Decode 按单请求维度执行（仅利用 GQA group size $\le 8$ 的查询并行），算术强度被锁死在 $\text{AI}_{\text{flat}} \approx 8.0\text{ FLOPs/Byte}$，完全受限于 HBM 带宽，导致共享前缀的物理 KV 页在每步 Decode 中被重复从 HBM 流式读取 $B$ 次。

---

## 2. 一手源码核验：`MultiLevelCascadeAttentionWrapper` 与 Base-2 LSE 归约

我们对 `flashinfer-ai/flashinfer`（`commit f64b4ba8762ded12a4f4b9a2092974c9a395e7ca`）进行了逐文件源码核验：

### 2.1 多级树形调度：`flashinfer/cascade.py`（blob `a1c7e2ddc4c11cc84bc3cf6afe3df3f5546a7baa`）
在 `MultiLevelCascadeAttentionWrapper`（第 421–574 行）中：
- **构造期（第 472–486 行）**：为前缀树的每一层 `level_i` 实例化一个 `BatchPrefillWithPagedKVCacheWrapper`。注意第 `0 .. num_levels - 2` 层（所有共享前缀层）由于整批 Query 都位于共享前缀 Token 之后，显式设置 `causal=False`，将原本内存受限的 Decode 转化为高吞吐、无因果掩码分支的 **Grouped-Query BatchPrefill (Dense Tile GEMM)**；仅在最后一层（叶子层 `num_levels - 1`，对应各请求独有后缀）保留 `causal=causal`。
- **执行期（`run()`，第 529–557 行）**：
  ```python
  out, lse = self._batch_prefill_wrappers[-1].run(q, paged_kv_cache, *args, return_lse=True)
  for batch_prefill_wrapper in self._batch_prefill_wrappers[:-1]:
      out_i, lse_i = batch_prefill_wrapper.run(q, paged_kv_cache, *args, return_lse=True)
      merge_state_in_place(out, lse, out_i, lse_i)
  ```
  先计算叶子层独立后缀的局部注意力输出 `(out, lse)`，随后自顶向下计算各共享前缀层 `(out_i, lse_i)` 并调用 `merge_state_in_place` 原地归约。

### 2.2 Base-2 对数和指数（LSE）状态合并与 `-inf` 防 NaN 保护：`include/flashinfer/attention/cascade.cuh`（blob `71ee5b7aee043417f3f415f826931d4d91fa6385`）
FlashInfer 在内核内部使用以 2 为底的对数和指数（$s = \log_2 \sum_j 2^{x_j \log_2 e}$）以直接映射 GPU SFU 的单指令 `ex2.approx` 与 `lg2.approx`。在 `MergeStateInPlaceKernel`（第 78–122 行）中：
```cuda
float s_a = s[s_offset];
float s_b = s_other[s_offset];
float s_max = max(s_a, s_b);
s_a = math::ptx_exp2(max(s_a - s_max, -math::inf));
s_b = math::ptx_exp2(max(s_b - s_max, -math::inf));
float d_sum = s_a + s_b;
float scale = d_sum > 0.f ? s_a / d_sum : 0.f;
float other_scale = d_sum > 0.f ? s_b / d_sum : 0.f;
// ...
v_a = scale * v_a + other_scale * v_b;
s[s_offset] = d_sum > 0.f ? math::ptx_log2(d_sum) + s_max : -math::inf;
```
- **空前缀段 `-inf` 边界防护**：当某个请求的某一级前缀为空（长度 0）时，该段的 LSE 为 $-\infty$。若两个段均为空（$s_a = -\infty, s_b = -\infty$），直接相减 $s_a - s_{\max} = (-\infty) - (-\infty)$ 在 IEEE-754 下会产生 `NaN`。FlashInfer 通过 `max(s_a - s_max, -math::inf)` 与 `d_sum > 0.f ? ... : 0.f` 严格屏蔽了 `NaN` 传播。
- **持久化多状态合并与 PDL（第 692–742 行）**：当需要将多个层级的状态一次性归约时，`VariableLengthMergeStates` 启动 `PersistentVariableLengthMergeStatesKernel`（采用 `num_smem_stages = 4` 的 `cp.async` 共享内存流水线），并支持 Hopper/Blackwell 的 `enable_pdl`（`cudaLaunchAttributeProgrammaticStreamSerialization`），让前序 Attention Kernel 尾波写回与 MergeState Kernel 前导加载在硬件级重叠。

---

## 3. 原创 CPU/Offline 实验与定量对比

实验脚本：`experiments/flashinfer_cascade_paged_kv/run.py`（结果固化于 `experiments/flashinfer_cascade_paged_kv/results.json`）。

### 3.1 Base-2 LSE 多级归约数值等价性核验
对 `level0_system = 12`、`level1_rag_group = 8`、`level2_unique_suffix = 5` 的 3 级前缀树执行 `compute_segment_attention_base2` 与 `merge_state_in_place_base2`，对比全序列拼接的 Monolithic Attention：
- Monolithic Base-2 LSE：`4.55102985` vs 3-Level Cascade Merged LSE：`4.55102985`（LSE 绝对误差 `0.0`）；
- 最终注意力输出向量最大绝对误差（Max Output Abs Error）：**`3.64e-17`**（机器双精度舍入极限）；
- 空段（`L=0`, `LSE=-inf`）合并误差：**`0.0`**，双空段返回 `-inf` 与全零向量验证通过。

### 3.2 Llama-3.1-70B (`TP=4`, 80 层) 批次规模与级联深度扫描
测试负载：每请求总上下文 `25,600` tokens（全批共享系统提示词 $L_{\text{sys}} = 16,384$，4 组共享 RAG 前缀 $L_{\text{rag}} = 8,192$，每请求独立后缀 $L_{\text{uniq}} = 1,024$）：

| Batch Size $B$ | Flat Paged-KV HBM 读取 (GiB) | 2-Level Cascade HBM (GiB) | 3-Level Cascade HBM (GiB) | HBM 流量削减 (%) | Flat 延迟 (ms) | 3-Level (无 PDL) (ms) | 3-Level + PDL 延迟 (ms) | 端到端加速比 | 算术强度跃迁 (FLOPs/B) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **16** | 31.260 | 12.515 | **5.020** | **83.94%** | 12.217 | 3.651 | **2.403** | **5.08x** | `8.0 -> 49.81` |
| **32** | 62.520 | 23.780 | **6.290** | **89.94%** | 23.994 | 4.130 | **2.882** | **8.33x** | `8.0 -> 79.50` |
| **64** | 125.039 | 46.310 | **8.829** | **92.94%** | 47.549 | 5.086 | **3.838** | **12.39x** | `8.0 -> 113.27` |
| **128** | 250.078 | 91.369 | **13.909** | **94.44%** | 94.657 | 7.000 | **5.752** | **16.46x** | `8.0 -> 143.81` |
| **256** | 500.156 | 181.488 | **24.067** | **95.19%** | 188.875 | 10.827 | **9.579** | **19.72x** | `8.0 -> 166.22` |

---

## 4. 关键架构师结论与反证边界

1. **算术强度从 Memory-Bound 向 Compute-Approach 跃迁**：在 $B=128$ 时，3-Level Cascade 将 80 层 Attention 的 HBM 读取量从 `250.078 GiB` 压降至 **`13.909 GiB`**（削减 **`94.44%`**），等效算术强度从 `8.0 FLOPs/Byte` 提升 **`18.0x`** 至 **`143.81 FLOPs/Byte`**，单步 Attention 延迟从 `94.657 ms` 降至 **`5.752 ms`**（提速 **`16.46x`**）。
2. **何时 Cascade Attention 会产生负收益（反证边界）？**
   - 当共享前缀极短（如 $L_{\text{sys}} < 256$ tokens）或共享分支数等于批大小（每个请求前缀均不相同，无复用度）时，拆分为多级 Kernel 并额外读写中间 `(V_partial, LSE)` 状态会引入每层 $3 \sim 4$ 次 Kernel Launch 开销及额外 HBM 往返；
   - 因此，调度器（如 vLLM V1 / SGLang RadixCache）必须设定最小共享前缀阈值（通常 $L_{\text{shared}} \ge 512$ 且共享请求数 $B_{\text{group}} \ge 4$）才激活 Cascade 分支。

---

## 5. 架构师练习与可证伪问题

1. **计算练习**：若将 KV Cache 从 BF16（2 Bytes/elem）量化为 FP8 E4M3（1 Byte/elem），保持 $B=128, L_{\text{sys}}=16384, L_{\text{rag}}=8192, L_{\text{uniq}}=1024$，计算 3-Level Cascade 的中间 MergeState HBM 读写流量占总 KV 流量的百分比变化。
2. **可证伪假说**：在 SM90/SM100 上开启 `enable_pdl=True` 对小批量（$B=16$）3-Level Cascade 的相对延迟改善比例（`34.2%`，从 `3.651 ms` 降至 `2.403 ms`）显著高于大批量（$B=256$ 时的 `11.5%`）。
