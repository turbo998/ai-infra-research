# `EAGLE-3` / DeepSeek `MTP` 树状投机解码验证、PD 首步拒绝采样与高并发屋脊线坍塌（Roofline Collapse）边界

> **证据类型声明**：本文源码分析属 `implementation`（固定提交的 `sgl-project/sglang` 一手源码），期望接受长度与算术强度屋脊线模型属 `own-offline`（[`experiments/eagle3_speculative_tree/run.py`](../../experiments/eagle3_speculative_tree/run.py)），不宣称物理 GPU 实测（`own-hardware`）。

---

## 1. 树状投机解码的核心悖论：为什么低并发提速 `2.6x` 的草稿树在高并发下反而降速？

投机解码（Speculative Decoding，如 `EAGLE-3` 与 DeepSeek-V3 `MTP`）之所以能在低并发（`BS=1–4`）下显著加速，是因为自回归 Decode 处于**深度 HBM 带宽受限区（Memory-Bandwidth Bound）**：每次读取 `9.2 GB/GPU` 的活跃模型权重只验证 `1` 个 Token，算术强度仅 `~2 FLOP/Byte`，远低于 H100 FP8 的屋脊线拐点（Ridge Point，约 **`229.6` 个有效 Query Token/步**）。

然而，当草稿验证从链式（`num_verify_tokens = 4..6`）扩展为树状（`num_verify_tokens = 32..64`），且服务并发度上升至 `BS = 16..32` 时，单步送入目标模型验证的**有效查询 Token 总数**变为：
$$Q_{\text{eff}} = \text{BS} \times N_{\text{verify}}$$
当 $\text{BS}=32, N_{\text{verify}}=32$ 时，$Q_{\text{eff}} = 1,024$，达到屋脊线拐点（`229.6`）的 **`4.46x`**！此时目标模型完全进入 **Tensor Core 算力受限区（Compute-Bound）**，单步验证耗时随 $N_{\text{verify}}$ 线性暴涨，而树中未被采纳的大量分支（如 32 个候选节点仅接受 `5.36` 个）全部转化为无效 FLOPs 与非因果树掩码带宽开销，导致端到端吞吐发生**屋脊线坍塌（Roofline Collapse）**。

---

## 2. 一手源码核验：SGLang `eagle_utils.py`、`eagle_disaggregation.py` 与 `adaptive_spec_params.py`

### 2.1 多步草稿 KV 槽位排布与非因果树掩码构造（`sglang/srt/speculative/eagle_utils.py`）
- **源码定位**：`sgl-project/sglang` (`commit 0f4aedfecc3e1e420292541af4e9da8909b9b4f7`, `python/sglang/srt/speculative/eagle_utils.py`, blob `149b7e56c174ad999e9748960f9805d6793373ea`, lines `70-300`)：
  1. **多步草稿 KV 缓冲视图唯一真源（`per_step_draft_out_cache_loc`）**：源码第 `70-91` 行强制断言 `out_cache_loc.shape[0] == batch_size * topk * num_steps`，并通过 `out_cache_loc.view(batch_size, topk, num_steps).permute(2, 0, 1).reshape(num_steps, -1)` 生成每一步 Draft Forward 与 DeepSeek-V4 压缩注意力后端共享的物理槽位切片。
  2. **Top-K 候选树剪枝与组织（`organize_draft_results`）**：源码第 `113-143` 行将多步累积得分展平后执行 `torch.topk(score_list, num_draft_token - 1, dim=-1)`，并对 `top_scores_index` 排序以保持树拓扑因果序，随后从 `ss_token_list` 中 `gather` 出最终送入目标模型验证的 `num_verify_tokens`（含 1 个根 `bonus_token`）。
  3. **三种树掩码内存模式（`TreeMaskMode`）**：源码第 `145-220` 行定义了 `FULL_MASK = 0`、`QLEN_ONLY = 1` 与 `QLEN_ONLY_BITPACKING = 2`。在 `FULL_MASK` 下，掩码张量大小高达 `seq_lens_sum * num_verify_tokens + num_verify_tokens^2 * bs` 字节；源码第 `186-190` 行专门指出，若每步全量 `fill_(True)` 会产生数百 MB 的无效 memset，因此仅在需要读取前缀掩码列时才填充，其余树单元格由 `sgl_build_tree_kernel_efficient` 直接覆写。

### 2.2 PD 分离架构下的首步精确拒绝采样恢复（`sglang/srt/speculative/eagle_disaggregation.py`）
- **源码定位**：`python/sglang/srt/speculative/eagle_disaggregation.py` (blob `f8b5a23bb69b9533ffdc2676c7f9a09062968d2a`, lines `26-147`)：
  - 在 Prefill-Decode 分离架构下，Prefill 节点向 Decode 节点移交请求时只传输了草稿 Token 索引 `topk_index`，而未传输体积庞大的完整词表分布 $q(x)$。
  - `pd_first_draft_proposal(topk_index, vocab_size)`（lines `26-47`）通过构造以 `topk_index` 为 `1.0` 的 One-Hot 提议分布 $q(X)=1$，使得 Decode 首步验证以概率 $p(X)$ 接受草稿，若拒绝则从剔除 $X$ 后的残差分布中重采样，从而在不跨网传输词表 Logits 的前提下**严格保持目标分布 $p$ 的无损精确性**；同时第 `84-108` 行在进入 Draft CUDA Graph 前将请求相对位置 `dsa_topk_indices` 一次性重映射为 Decode 本地物理槽位（保留槽位 `0` 作为 Padding Sink）。

### 2.3 基于 Batch Size 与 EMA 接受长度的自适应降阶状态机（`sglang/srt/speculative/adaptive_spec_params.py`）
- **源码定位**：`python/sglang/srt/speculative/adaptive_spec_params.py` (blob `268ceaeafa77fc59215dd233f26211ed8d423311`, lines `27-52`, `143-348`)：
  - SGLang 在源码第 `27-52` 行直接硬编码了随 Batch Size 阶梯式收缩的 `DEFAULT_ADAPTIVE_CONFIG`：
    - `"1": {"candidate_steps": [1, 3, 5, 7], "down_hysteresis": -0.25}`
    - `"8": {"candidate_steps": [0, 1, 3]}`
    - `"32": {"candidate_steps": [0, 1]}`
    - `"64": {"candidate_steps": [0]}`（当 `BS >= 64` 时直接置 `steps=0` 关闭投机解码！）
  - `AdaptiveStepSlot` 维护指数移动平均接受长度 `self.ema_accept_len = (1 - alpha) * ema + alpha * batch_avg`（默认 `ema_alpha=0.2`，每 `5` 个 Batch 更新一次），当 `ema_accept_len <= prev_step - 0.5 + down_hysteresis` 时自动降档，并在路由时先通过 `_pad_to_cuda_graph_bs(batch_size)` 对齐到 CUDA Graph 捕获桶。

```
Roofline Collapse Mechanism in Tree Speculative Decoding (H100 FP8 Ridge Point = 229.6 Query Tokens):
  BS = 1,  N_verify = 32  ==> Q_eff =   32 (0.14x Ridge, Memory-Bound)  ==> E[tau] = 5.364, Speedup = 2.605x ✅
  BS = 32, N_verify = 32  ==> Q_eff = 1024 (4.46x Ridge, Compute-Bound) ==> E[tau] = 5.364, Speedup = 0.889x ✕ (Collapsed!)
  BS = 32, Adaptive Chain ==> Q_eff =  192 (0.84x Ridge, Balanced)      ==> E[tau] = 3.815, Speedup = 2.422x ✅ (Recovered!)
```

---

## 3. 离线实验结果（`experiments/eagle3_speculative_tree/`）

| 场景 ($\alpha$) | Batch Size | 策略配置 | $N_{\text{verify}}$ | 有效查询 Token $Q_{\text{eff}}$ | 屋脊线饱和度 ($Q_{\text{eff}} / Q_{\text{ridge}}$) | 期望产出 $\mathbb{E}[\tau]$ | 单步耗时 | 净加速比 (vs NoSpec) |
|---|---|---|---|---|---|---|---|---|
| `code_math` (`0.86`) | `1` | `tree_s6_k8_v64` | `64` | `64` | `0.279x` | `6.142` | `10.295 ms` | **`2.647x`** |
| `code_math` (`0.86`) | `1` | `tree_s5_k6_v32` | `32` | `32` | `0.139x` | `5.364` | `9.124 ms` | **`2.605x`** |
| `code_math` (`0.86`) | `32` | `tree_s5_k6_v32` (固定宽树) | `32` | `1,024` | **`4.460x`** | `5.364` | `47.349 ms` | **`0.889x` ✕ (坍塌)** |
| `code_math` (`0.86`) | `32` | **`chain_s5_k1_v6` (自适应降阶)** | **`6`** | **`192`** | **`0.836x`** | **`3.815`** | **`12.368 ms`** | **`2.422x` ✅ (恢复)** |
| `general_chat` (`0.74`) | `16` | `tree_s6_k8_v64` (固定宽树) | `64` | `1,024` | **`4.460x`** | `4.677` | `44.782 ms` | **`0.660x` ✕ (严重退化)** |
| `high_entropy` (`0.58`) | `64` | `tree_s5_k6_v32` vs `nospec` | `32` vs `1` | `2,048` vs `64` | `8.920x` vs `0.279x` | `3.153` vs `1.000` | `89.305 ms` vs `11.384 ms` | `0.402x` vs **`1.000x` (关闭投机)** |

---

## 4. 架构师设计准则与可证伪问题

1. **严禁在生产高并发池硬编码静态大草稿树（`num_verify_tokens >= 32`）**：必须遵循 $Q_{\text{eff}} = \text{BS} \times N_{\text{verify}} \le 0.85 \times Q_{\text{ridge}}$ 的屋脊线预算准则，随着并发度上升动态将树宽从 `64 -> 32 -> 16 -> 6 (Chain) -> 1 (NoSpec)` 收缩。
2. **模式耦合禁令**：正如 `adaptive_spec_params.py` (lines `70-89`) 所揭示，启用 Attention DP（`attn_dp_enabled`）、双批次重叠（`enable_two_batch_overlap`）或 PDMux 时，不同 Rank 若各自根据本地 EMA 切换 `num_steps` 会导致集合通信步数失步死锁，此时必须在 DP Group 内同步自适应档位或回退至固定小步长链式 MTP。
