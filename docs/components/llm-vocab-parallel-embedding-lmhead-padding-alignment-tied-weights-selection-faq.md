# 大模型词表并行（`VocabParallelEmbedding / lm_head`）`TP × 128` 硬件对齐 Padding、幽灵词（`Ghost Tokens`）`-inf` 掩码与流水线共享词表梯度同步选型：为什么 `Qwen2.5 / Qwen3` 的分词器只有 `151,665` 个词，模型 `config.vocab_size` 却要硬补 `399` 行变成 `152,064`（让 `lm_head` 提速 `1.63x`）？如果推理时忘了对这 `399` 个幽灵行屏蔽 `-inf` 或在 `PP=8` 下漏掉首尾 Stage 共享梯度同步会引发什么事故？

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/run.py` 与 `results.json`。

![Vocab-Parallel Padding Alignment & Tied-Weight Sync Visual Summary](../../assets/svg/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向大词表张量并行与流水线并行（`Megatron-Core make_vocab_size_divisible_by=128 / vLLM VocabParallelEmbedding / TorchTitan`）的“每 TP 分片 `128` 元素对齐填充（`V_pad = ceil(V_raw / (TP * 128)) * (TP * 128)`）+ 越界幽灵行（`[V_raw : V_pad]`）零初始化与采样 `-inf` 硬屏蔽 + 跨流水线首尾 Stage `embedding_group` 共享权重梯度 `All-Reduce`”全链路规范**。
- **架构师价值**：很多工程师在加载开源大模型（尤其是 `Qwen2.5 / Qwen3` 或自定义扩充词表后的微调模型）时，都会注意到一个让人困惑的细节：**调用 `len(tokenizer)` 返回的真实词表大小明明是 `151,665`（或 `151,936`），为什么 `config.json` 里的 `vocab_size` 却写着 `152,064`，凭空多出了 `399` 个根本没有对应文字的“幽灵 Token（Ghost Tokens）”？**
  围绕这多出来的几百行词表，在训练和推理引擎中潜伏着三大经典线上事故：
  1. **事故一：微调或自训新词表时，直接拿未对齐的 `len(tokenizer) = 151,665` 作为 `Embedding / lm_head` 维度，不仅无法被 `TP=8` 整除，还让 `lm_head` GEMM 退化到标量慢速路径（耗时从 `2.98 ms` 暴涨至 `4.85 ms`，慢了 `1.63 倍`，Tier 1）**！
     - **硬件机理**：在 NVIDIA Hopper / Blackwell 架构中，`WGMMA` 指令、异步张量内存加速器（`TMA` 128-Byte 对齐）以及 `FP8 / NVFP4` 块级缩放（`128×128` Scaling Tile）都要求矩阵的 $M, N, K$ 维度（在每个 `TP` Rank 分片之后！）必须是 **`64` 或 `128` 个元素的整数倍**！
     - 如果词表大小是奇数 `151,665`（甚至虽然能被 8 整除但除完不是 128 的倍数），`cuBLASLt` 无法使用 16 字节向量化加载（`LDG.E.128`）和 TMA 硬件拷贝，被迫回退到未对齐边界谓词（Predicated Unaligned Path），导致占据全模型参数量 `10%~20%` 的 `lm_head` 前反向矩阵乘利用率从 `78.6%` 暴跌至 **`48.2%`（耗时增加 `62.8%`）**！
     - 而把 `151,665` 向上补齐到 **`152,064`** 后，除以 `TP = 8` 正好等于每张卡 **`19,008` 行**，而 `19,008 = 148.5 × 128 = 297 × 64`（若按 `TP=8 × 128 = 1,024` 步长对齐，如 `152,576` 或 `TP=4` 下 `152,064 / 4 = 38,016 = 297 × 128`）完美对齐 Tensor Core 硬件 Tile！
  2. **事故二：推理引擎加载了 `152,064` 行的权重，但采样前忘了把最后 `399` 行幽灵词（下标 `151,665 .. 152,063`）的 Logits 置为 `-inf`，导致线上高温度采样（`temperature >= 1.0`）时有 `1.80%` 的请求抽中幽灵 Token ID，触发反分词器 `KeyError / IndexError` 当场崩溃（Tier 2）**！
     - 在模型刚初始化或经过 SFT/RL 训练（尤其是加了 Weight Decay 或噪声）后，下标 `[151,665 : 152,064)` 这 399 行幽灵权重的向量并不严格全等于负无穷，它们算出来的 Logits 往往是 `0.0`（而正常罕见词的 Logits 可能是 `-8.0`）！
     - **Logit 为 `0.0` 比 `-8.0` 大得多**！如果不显式把 `[V_raw : V_pad)` 的 Logits 屏蔽为 `-inf`，采样器就会以高达 **`1.80%`** 的概率抽中比如 `token_id = 151,888`，随后送到 `tokenizer.decode([151888])` 时，因为分词器字典里最多只有 `151,664`，直接返回 `None` 或抛异常崩掉！
  3. **事故三：在小参数模型（如 `1B / 3B / 7B` 开启 `tie_word_embeddings=True` 词表与输出头共享权重）使用流水线并行（`PP = 8`）训练时，忘了在 Stage `0` 和 Stage `7` 之间做 `All-Reduce(embedding_grad)`，导致首尾两端的“共享权重”越训越不一样，验证集困惑度（PPL）恶化 `+0.42`（Tier 3）**！
  采用 **向上对齐至 `TP × 128` 整数倍 + 在最后一个 TP Rank 上将 `[V_raw : V_pad)` 的幽灵行权重置零且在 Logits 上硬屏蔽 `-inf` + 在 `PP > 1` 且 `tie_word_embeddings=True` 时建立跨首尾 Stage 的 `embedding_group` 同步梯度（Tier 4）**，可在仅增加 **`0.26%`** 词表显存的前提下，让 `lm_head` GEMM 提速 **`1.63 倍`（`4.85 ms -> 2.98 ms`）**，并将越界幽灵词崩溃率与共享权重漂移双双清零（**`0.0%`**）！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：词表并行（`VocabParallel`）的硬件对齐公式**：
   - 设分词器真实词表大小为 $V_{\text{raw}} = \text{len}(\text{tokenizer})$，最大张量并行度为 $T = \text{tp\_size}$，单卡 Tensor Core / FP8 块对齐粒度为 $M = 128$（即 Megatron 中的 `--make-vocab-size-divisible-by 128`）。则模型实际分配的填充词表大小 $V_{\text{pad}}$ 为：
     $$V_{\text{pad}} = \left\lceil \frac{V_{\text{raw}}}{T \times M} \right\rceil \times (T \times M), \qquad V_{\text{local}} = \frac{V_{\text{pad}}}{T} \equiv 0 \pmod{128}$$
   - 在 `TP` 切分（按行切分词表 `Row-Sharded Embedding` / `ColumnParallel lm_head`）下，第 $r \in \{0..T-1\}$ 张卡负责的词表闭开区间为：
     $$\bigl[\text{start}_r, \;\text{end}_r\bigr) = \bigl[r \cdot V_{\text{local}}, \;\;(r+1) \cdot V_{\text{local}}\bigr)$$
2. **第一性原理：幽灵词行的“双重防御法则（Zero-Init + Logit `-inf` Masking）”**：
   - 由于所有的 $N_{\text{ghost}} = V_{\text{pad}} - V_{\text{raw}}$ 个幽灵词全都落在了最后一个 TP 分片（或最后几个分片）的末尾，必须在两处设防：
     1. **前向 `VocabParallelEmbedding` 越界掩码**：当输入 Token ID $x \notin [\text{start}_r, \text{end}_r)$ 时，先生成掩码 `mask = (x < start_r) | (x >= end_r)`，将越界的局部下标置为 `0` 查表后再把对应输出行乘以 `0.0`，最后做 `TP All-Reduce(SUM)`；
     2. **输出 `lm_head` 幽灵列 `-inf` 屏蔽**：在算完 `logits = hidden @ W_lm_head.T` 后，对于所有全局词表下标 $v \ge V_{\text{raw}}$ 的列，**在进入 Softmax / Cross-Entropy / Sampler 之前强制执行 `logits[..., v >= V_raw] = -inf`**！这样既不会在训练时向幽灵行泄漏任何 Softmax 分母概率质量，又能在推理时 `100%` 杜绝采样出幽灵 Token！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果在 SFT 微调时，用户往分词器里用 `tokenizer.add_special_tokens()` 新增了 3 个自定义特殊 Token，调用 `model.resize_token_embeddings(len(tokenizer))` 会发生什么惨剧？
- **这是 HuggingFace 微调里最隐蔽的“性能腰斩陷阱”！**
  原本模型（如 `Qwen2.5`）的 `Embedding` 矩阵已经精心对齐到了 `152,064` 行，其实里面早就预留了 `399` 个空闲槽位（`151,665 .. 152,063`）专门给你放新增 Token！
  如果你不知道这件事，在加了 3 个词（`len(tokenizer)` 变成 `151,668`）后盲目调用了不带对齐参数的 `model.resize_token_embeddings(151668)`，HuggingFace 会 **把原本对齐的 `152,064` 行矩阵截断缩容成不伦不类的 `151,668` 行**！不仅直接毁掉了 `128` 元素 Tensor Core 对齐（让训练慢 `30%+`），而且在导出到 `vLLM` 跑 `TP=8` 时因为 `151,668 % 8 = 4 != 0` 直接报错无法加载！
  - **正确做法**：如果 `len(tokenizer) <= model.config.vocab_size`，**根本不需要调用 `resize_token_embeddings`**！即便超出，也必须传 **`model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=1024)`**！

### Q2：为什么在 `PP > 1`（流水线并行）下，`tie_word_embeddings=True`（输入词嵌入与最后一层 `lm_head` 共享同一份权重）需要专门建一个跨首尾 Stage 的 `embedding_group`？
- 因为在普通单卡或纯 `TP/DP` 中，`wte.weight` 和 `lm_head.weight` 在同一个 GPU 进程内指向同一个物理显存指针（`lm_head.weight = wte.weight`），反向传播时两者的梯度会自动加到同一个 `main_grad` 里；
- 但在 **流水线并行（如 `PP = 8`）** 中：
  - 输入层 `wte` 位于 **Stage `0`（第 1 台机器）**；
  - 输出层 `lm_head` 位于 **Stage `7`（第 8 台机器）**！
  两张卡隔着网络，各自在本地保存了一份该权重的副本！如果在每个大 Step 结束、`optimizer.step()` 执行之前，没有在一个僅包含 `(Stage 0, Stage 7)`（若开启了交错虚拟流水线还可能包含相应虚拟 Stage）的 **`embedding_group` 上执行一次 `dist.all_reduce(w.main_grad)`**，那么 Stage 0 只用输入层梯度更新，Stage 7 只用输出层梯度更新——10 个 Step 之后两边的权重就彻底分道扬镳了！

### Q3：为什么现在 `14B / 32B / 70B / 671B` 大模型几乎全都改成了 `tie_word_embeddings=False`（解耦输入 Embedding 与 `lm_head`）？
- 因为对于百亿/千亿模型，词表参数只占总参数的 `1% ~ 3%`，省这点显存根本不划算，反而会带来流水线首尾 Stage 的额外跨网 `All-Reduce` 同步开销，且输入词空间与输出分类空间在分布范数上本就有差异，解耦后收敛更好、架构更干净！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 大模型词表并行对齐、幽灵行屏蔽与共享权重同步选型决策树

```mermaid
flowchart TD
    A["训练或部署大词表模型 (分词器真实词数 V_raw = 151,665, TP=8, PP=8)"] --> B{"模型词表维度 V_pad 是否对齐至 TP × 128 的整数倍?"}
    B -- "否: 直接使用 V_raw = 151,665" --> C["⛔ Tier 1: 无法被 TP=8 整除, 且错过 Tensor Core 128 元素对齐 -> lm_head 慢 1.63x (4.85 ms)!"]
    B -- "是: 向上补齐至 V_pad = 152,064 (每卡 19,008 行 = 148.5 × 128)" --> D{"是否将多出的 399 行幽灵词 [151,665 : 152,064) Logits 屏蔽为 -inf?"}
    D -- "否: 未做幽灵列 -inf 掩码" --> E["⛔ Tier 2: 幽灵行 Logit=0 高于正常罕见词 (-8.0) -> 1.80% 概率抽中越界 ID 引发反分词器崩溃!"]
    D -- "是: 幽灵列强制置 -inf" --> F{"若开启 tie_word_embeddings=True 且 PP > 1, 首尾 Stage 是否同步梯度?"}
    F -- "否: Stage 0 与 Stage 7 各自独立更新" --> G["⛔ Tier 3: 首尾共享权重分道扬镳 -> 验证集 PPL 恶化 +0.42!"]
    F -- "是: 建立首尾 Stage embedding_group 执行 All-Reduce(main_grad)" --> H["🏆 Tier 4: lm_head 2.98 ms (快 1.63x) + 0.0% 幽灵词崩溃 + 0.00 PPL 漂移!"]
```

### 3.2 未对齐词表与幽灵词泄露 vs. `TP × 128` 硬件对齐 + `-inf` 掩码对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 未对齐维度拖慢 GEMM (4.85ms) + 未屏蔽幽灵行引发 1.8% 崩溃!"]
      B1["V_raw = 151,665: 错失 128-elem TMA/FP8 对齐 -> 4.85 ms"] --> B2["补齐至 152,064 但未屏蔽 [151665:152064] -> 抽中 ID 151,888 导致 tokenizer.decode 崩溃!"]
    end
    subgraph Good["🏆 Tier 4: 补齐至 152,064 (19,008/卡) + 幽灵列 -inf 屏蔽 + PP 首尾同步"]
      G1["V_pad = 152,064: 每卡 19,008 行完美对齐 Tensor Core -> 2.98 ms (快 1.63x!)"] --> G2["最后一个 TP Rank 将后 399 列置为 -inf -> 0.0% 越界采样, 100% 数学等价!"]
    end
```

### 3.3 `VocabParallel` 前向查表掩码、`lm_head` 幽灵列屏蔽与 `PP` 首尾梯度同步时序图

```mermaid
sequenceDiagram
    participant S0 as PP Stage 0 (VocabParallelEmbedding)
    participant S7 as PP Stage 7 (ColumnParallel lm_head, TP Rank 7)
    participantSampler as Softmax / Cross-Entropy / Sampler

    S0->>S0: 1. Forward lookup: zero-out tokens outside `[r*19008, (r+1)*19008)` & `All-Reduce(SUM)`
    S0->>S7: 2. Pipeline forward activation `H` through Stages `0 -> 7`
    S7->>S7: 3. Compute aligned GEMM `logits_local = H @ W_local.T` (19,008 cols, 2.98 ms)
    S7->>Sampler: 4. Mask ghost columns `logits_local[:, 151665 - 7*19008 :] = -inf` (399 cols)
    S7->>S0: 5. Backward complete -> `dist.all_reduce(wte.main_grad, group=pp_embedding_group)`!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 词表对齐、幽灵词掩码与 PP 共享同步方案 | `lm_head` GEMM 耗时 | Tensor Core 利用率 | 越界幽灵词崩溃率 | `PP` 共享权重 PPL 劣化 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 直接使用原始词表 (`151,665`)** | 未做 `TP × 128` 补齐，标量边界回退 | `4.85 ms` ✕ | `48.2%` ✕ | `0.0%` | `0.00` |
| **Tier 2: 补齐至 `152,064` (未屏蔽幽灵行)** | 对齐加速，但 `[151665:152064]` 未置 `-inf` | `2.98 ms` | `78.6%` | **`1.80%` (线上崩溃! ✕)** | `0.00` |
| **Tier 3: 补齐 + 幽灵掩码 (漏掉 PP 首尾同步)** | 推理正常，但 `PP=8` 下首尾共享权重未归约 | `2.98 ms` | `78.6%` | `0.0%` | **`+0.42` (掉点! ✕)** |
| **Tier 4: `TP × 128` 补齐 + `-inf` 掩码 + PP 同步 (推荐)** | **128 对齐 + 幽灵列 `-inf` + 首尾梯度同步** | **`2.98 ms` (`1.63x` 🏆)** | **`78.6%` (`+30.4 pt` 🏆)** | **`0.0%` 🏆** | **`0.00` (无损 🏆)** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/run.py`](../../experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-sizer.html`](../reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-selection-cards.html`](../reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unpadded Odd Vocab (151,665)", "gemm_ms": 4.85, "tc_util_pct": 48.2 },
      { "tier": "Tier 2: Padded 152,064 (Unmasked Ghost)", "gemm_ms": 2.98, "tc_util_pct": 78.6 },
      { "tier": "Tier 3: Masked + Unsynced Tied PP", "gemm_ms": 2.98, "tc_util_pct": 78.6 },
      { "tier": "Tier 4: TPx128 Padded + Masked + PP Sync", "gemm_ms": 2.98, "tc_util_pct": 78.6 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "gemm_ms": "Quantity",
    "tc_util_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "TPx128 Vocabulary Padding (151,665 -> 152,064) Speeds Up lm_head GEMM 1.63x (4.85ms -> 2.98ms)",
    "subtitle": "lm_head GEMM latency (ms) across 4 vocabulary alignment tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "gemm_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有大词表模型（`Qwen2.5/3`, `Llama-3`, `DeepSeek-V3`）的预训练、SFT 扩词表微调（`resize_token_embeddings`）与 `vLLM / SGLang` 张量并行推理。
- **⛔ Not for**：直接把奇偶不齐的 `len(tokenizer)` 设为模型词表维度，或对填充行 `[len(tokenizer) : config.vocab_size)` 不做 `-inf` 屏蔽。

### 生产核验 Checklist
- [ ] **确保 `config.vocab_size` 始终对齐到 `TP_max × 128`（如 `1,024`）的整数倍**；调用 `resize_token_embeddings` 时强制传入 `pad_to_multiple_of=1024`。
- [ ] **在输出层对所有满足 `vocab_idx >= len(tokenizer)` 的填充幽灵列强制执行 `logits[..., vocab_idx] = -inf`**。
- [ ] **当 `tie_word_embeddings=True` 且 `PP > 1` 时，在 `optimizer.step()` 前对流水线首尾 Stage 的词表梯度执行 `All-Reduce` 同步**。
