# 代码大模型中间补全（`FIM: SPM vs PSM`）与 BPE 词边界回溯治愈（`Token Healing`）选型：为什么程序员把光标停在半截变量名（`def calculate_tot|`）或 URL（`https://|`）时，BPE 分词断裂会让代码补全准确率暴跌 `-18.4 pt`（`76.2% -> 57.8%`），而用 `PSM` 格式又会让每次敲键盘时后缀缓存全部失效？基于 `SPM`（后缀前置）格式 + 末位 Token 回溯与字典树前缀掩码治愈（`Token Healing`，准确率拉升至 `76.2%`，缓存命中率 `91.4%`，TTFT 提速 `4.25x`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_code_fim_token_healing_bpe_prefix_boundary_backtracking_selection/run.py` 与 `results.json`。

![Code LLM FIM (SPM vs PSM) & BPE Token Healing Visual Summary](../../assets/svg/llm-code-fim-token-healing-bpe-prefix-boundary-backtracking-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向 IDE 实时代码补全服务（`Qwen2.5-Coder / DeepSeek-Coder / StarCoder2` 配合 `vLLM / SGLang APC`）的“`SPM`（`Suffix-Prefix-Middle`）后缀前置缓存最大化格式 + 末位 1-Token BPE 边界回溯（`Token-Boundary Backtracking`）+ 首步 `Marisa-Trie` 词表前缀位图掩码治愈（`Token Healing`）”全链路架构**。
- **架构师价值**：在用开源代码大模型（`Qwen2.5-Coder-32B`, `DeepSeek-Coder-V2`）构建企业内部类 Copilot / Cursor 的 IDE 实时灰色幽灵字补全（Inline Ghost-Text Completion）服务时，工程团队往往会撞上两个极其困惑的痛点：**① 当光标停在空格或换行后时补全很准，但只要程序员在敲一个长变量名或 URL 的半中间停下来（比如刚敲到 `def calculate_tot` 或 `url = "https://`），模型就突然变傻，要么漏掉后半截单词直接吐括号 `(items):`，要么吐出三个斜杠 `https:///github.com`（准确率暴跌 `-18.4 pt`，从 `76.2%` 跌至 `57.8%`）！② 程序员每敲一个字母，GPU 的自动前缀缓存（`APC`）命中率只有可怜的 `42.0%`，P99 首字延迟 `TTFT` 高达 `48.5 ms`！**
  拆解 BPE 分词器与 FIM（Fill-in-the-Middle）格式的第一性原理，这两大痛点的根源一目了然：
  1. **根源一：BPE 贪心分词在光标断词处产生的“分布外死胡同（Out-of-Distribution Token Split，掉点 `-18.4 pt`，Tier 1 & Tier 2）”**！
     - 在代码模型的 BPE 词表中，常见完整词根如 `"_total"` 是一个 **单独的原子 Token `[14892] ("_total")`**！
     - 在模型预训练读过的所有百亿行代码里，凡是出现 `calculate_total` 的地方，分词序列永远是 `["calculate", "_total"]`，**模型一辈子也从来没见过 `["calculate", "_", "tot", "al"]` 这种拆法**！
     - 然而，当用户在 IDE 里刚敲到 `"def calculate_tot"` 停手等待补全时，分词器看不到后面的 `"al"`，只能被迫把末尾编码成 **`[62 ("_"), 19402 ("tot")]`** 发给模型！
     - 模型一看输入结尾是独立的 Token `"tot"`，根据预训练先验，它认为变量名已经写完了（因为如果要写 `_total`，前一个词就绝不可能是 `"tot"`！），于是直接跳过 `"al"` 吐出了左括号 `"("`！
     - 同理，当用户敲完 `"https://"` 时，分词器将其切为 `["https", "://"]`；而在正常 URL `"https://github.com"` 中，BPE 的贪心合并规则是把双斜杠和域名合成 **`["https", ":", "//github", ".com"]`**！因为输入末尾已经是 `"://"`，模型为了接上 `"//github"` 就会多吐一个斜杠变成 **`https:///github.com`**！
  2. **根源二：使用 `PSM`（`Prefix-Suffix-Middle`）格式导致每次敲击键盘时，光标下方的 `2,000` Token 后缀 KV Cache 100% 移位失效（命中率仅 `42.0%`，Tier 1）**！
     - 代码模型在训练时都同时支持两种 FIM 排序：
       - **`PSM`**：`<|fim_prefix|> {光标前代码 P} <|fim_suffix|> {光标后代码 S} <|fim_middle|>`
       - **`SPM`**：`<|fim_prefix|><|fim_suffix|> {光标后代码 S} <|fim_middle|> {光标前代码 P}`
     - 想想程序员在 IDE 里是怎么写代码的：**随着手指不断敲字，光标上方的 `{P}` 末尾每 100 毫秒都在变长，而光标下方的整篇文件后半部分 `{S}`（通常有 `2,048` 个 Token）是纹丝不动的**！
     - 如果你用 **`PSM`** 格式，把每天都在变的 `{P}` 放在前面、把不变的 `{S}` 放在后面：用户每多敲 1 个字母，`{P}` 的长度就 `+1`，导致后面整整 `2,048` 个 Token 的 `{S}` 在序列中的绝对位置全部往后挪了 1 位——**`{S}` 的所有 PagedAttention KV Cache 块在哈希树里 100% 全部失效重算**！而且在 `PSM` 中，`{P}` 和生成的 `{Middle}` 中间还隔着整段 `<|fim_suffix|>{S}<|fim_middle|>`，导致 `{P}` 的最后一个词根本无法与生成的第一个词做 Token Healing！
  采用 **`SPM`（后缀前置）格式（把不变的 `{S}` 放在最前面，把 `{P}` 放在紧挨着生成区 `<|fim_middle|>` 的最后面，使 GPU 前缀缓存命中率从 `42.0%` 飙升至 `91.4%`，`TTFT` 从 `48.5 ms` 缩至 `11.4 ms`，提速 `4.25 倍`）+ 1-Token BPE 回溯与 `Marisa-Trie` 词表前缀掩码治愈（`Token Healing`，耗时仅 `0.08 ms`，将断词补全准确率从 `57.8%` 拉升至 `76.2%`，提升 `+18.4 pt`）（Tier 4）**，即可彻底解决上述两大顽疾！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：1-Step BPE 词边界治愈算法（The Token Healing Algorithm）**：
   - 当光标停在非空白字符末尾（如 `P = "def calculate_tot"`）时，在 `SPM` 格式下，`input_ids` 的最后一个 Token 是 $t_{\text{last}} = 19402$（对应字符串片段 $s_{\text{tail}} = \text{"tot"}$）。
   - **三步零损治愈**：
     1. **回退一步（Backtrack 1 Token）**：将最后一个 Token $t_{\text{last}}$ (`"tot"`) 从发送给 GPU 的 `input_ids` 末尾弹出（如果倒数第二个是标点如 `"://"` 则回退到标点前）；
     2. **首步前缀字典树掩码（Step-1 Trie Prefix Mask）**：在生成第 1 个 Token 时，在预构建的内存 **`Marisa-Trie`（词表前缀树）** 中以 `0.08 ms` 查出词表 $V$ 中所有 **以字符串 `"tot"` 为前缀** 的合法 Token ID 集合 $\mathcal{V}_{\text{prefix}(\text{"tot"})}$（例如 `["total", "totally", "tot", "tote", ...]`），将其他非前缀 Token 的 Logit 掩码为 `-inf`；
     3. **采样并剥离已敲前缀（Sample & Strip Prefix）**：此时模型在完全符合预训练分布的上下文下，高概率一把采样出完整 Token `"total"`；服务端将用户已经敲过的 `"tot"` 剥离掉（`"total"[3:] == "al"`），把剩余的 **`"al"`** 连同后续正常生成的 `"_cost(items):"` 流式返回给 IDE！
2. **第一性原理：为什么 `SPM` 是唯一同时成就“91.4% 缓存命中率”与“无缝 Token Healing”的 FIM 格式？**
   - 在 `SPM`（`<|fim_prefix|><|fim_suffix|> {S} <|fim_middle|> {P}`）下：
     1. **缓存最优**：排在最前面的 `{S}`（`2,048` Token）和 `{P}` 的前 $N-1$ 个 `16-Token` 块在连续敲字期间 `100%` 不变，每次敲击键盘只需要增量 Prefill 最后那 1 个半满块（`< 16` Token），`TTFT` 仅 **`11.4 ms`**！
     2. **边界毗邻**：`{P}` 的最后一个 Token 在物理位置上紧贴着生成的 `{Middle}` 第 1 个 Token，使得回退 `{P}` 的最后 1 个 Token 并在生成第 1 步做前缀掩码在数学上 **100% 严格等价于连续无断词生成**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果用户在敲代码的同时，每隔几秒钟也在往下滚动或修改文件下半部分（导致 `{S}` 偶尔变化），`SPM` 的缓存策略怎么设计最稳？
- **对 `{S}` 做固定块对齐截断（Block-Aligned Suffix Window）**：
  截取光标下方接下来的固定 `1,024` 或 `2,048` Token 作为 `{S}`，或者以光标所在函数之后的下一个顶层函数/文件末尾作为静态锚点（Anchor）：只要用户的修改发生在当前函数内部的光标处，下方的锚点函数块就纹丝不动，始终保持 `> 90%` 的前缀缓存命中率！

### Q2：什么时候需要回退 2 个 Token（2-Token Backtracking）而不是 1 个 Token？
- 当最后 1 个 Token 只有 1~2 个字符且前面紧挨着另一个子词或运算符碎片时（例如 `"https://"` 被切成 `["https", "://"]`，而目标词是 `[":", "//github"]`）：
  - 标准启发式规则是：**从 `input_ids` 末尾往前弹出 $k \in \{1, 2\}$ 个 Token，直到停在稳定的空白符/语法分界点，或弹出的尾部子串长度 $|s_{\text{tail}}| \ge 3$**，然后在第 1 步（必要时第 2 步）应用 Trie 前缀约束掩码！

### Q3：`Qwen2.5-Coder`、`DeepSeek-Coder` 和 `StarCoder2` 的 `SPM` 特殊 Token 拼接顺序完全一样吗？
- **注意细微差别（必须严格对齐各模型预训练时的 SPM 模板）**！
  - 在 `StarCoder2 / Qwen2.5-Coder` 中，标准 `SPM` 格式是：
    `<|fim_prefix|><|fim_suffix|>{S}<|fim_middle|>{P}`（注意最开头先放一个空的 `<|fim_prefix|>` 再放 `<|fim_suffix|>`，这样模型一开头就知道进入了 FIM 模式！）；
  - 切记必须把 `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` 作为不可拆分的特殊 Token（或 `add_special_tokens`）编码，绝不能让它们与 `{S}` 或 `{P}` 的首字符发生 BPE 合并！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 代码大模型 IDE 实时补全 `FIM` 格式与 `Token Healing` 选型决策树

```mermaid
flowchart TD
    A["IDE 实时代码补全请求 (光标前代码 P='...def calculate_tot', 光标后代码 S=2,048 Tokens)"] --> B{"FIM 提示词组织格式 (PSM vs SPM)?"}
    B -- "PSM: <|fim_prefix|> P <|fim_suffix|> S <|fim_middle|>" --> C["⛔ Tier 1: 每次敲字 P 变长导致后面 2,048 词的 S 位置全变 -> 缓存命中率仅 42%, TTFT 48.5 ms!"]
    B -- "SPM: <|fim_prefix|><|fim_suffix|> S <|fim_middle|> P" --> D{"是否对 P 末尾的半截标识符/符号执行 BPE Token Healing?"}
    D -- "否: 直接将 [..., '_', 'tot'] 送入模型生成" --> E["⛔ Tier 2: 虽缓存命中率达 91.4% (11.3 ms), 但因预训练只见过 '_total', 断词补全准确率仅 60.4%!"]
    D -- "仅按空格简单截断 (无词表 Trie 掩码)" --> F["⚠️ Tier 3: 丢弃了用户已敲的 'tot' 上下文约束, 容易补出无关变量名 (准确率 66.5%)"]
    D -- "弹出末尾 Token '_tot' + Step-1 施加 Marisa-Trie '_tot*' 前缀掩码 + 剥离已敲前缀" --> G["🏆 Tier 4: 断词准确率跃升至 76.2% (+18.4 pt) + 缓存命中率 91.4% + P99 TTFT 11.4 ms!"]
```

### 3.2 未治愈 `PSM` 断词失真（`57.8%` 准确率）vs. `SPM` + `Trie Token Healing`（`76.2%`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: PSM 格式 + 未治愈 BPE 断词 (准确率仅 57.8%, TTFT 48.5 ms!)"]
      B1["输入末尾 'calculate_tot' 被切为 ['calculate', '_', 'tot']"] --> B2["模型从未见过 'tot' 后接 'al' (只见过单 Token '_total') -> 错误吐出 '(items):'!"]
      B2 --> B3["PSM 下每次敲字使后缀 S 缓存全失效 -> TTFT 高达 48.5 ms!"]
    end
    subgraph Good["🏆 Tier 4: SPM 后缀前置格式 + 1-Token 回溯与 Marisa-Trie 前缀掩码治愈"]
      G1["SPM 将不变的 S 放最前 -> 91.4% 命中 GPU 前缀缓存 (TTFT 仅 11.4 ms, 快 4.25x!)"] --> G2["弹出末尾 '_tot', 在 Step 1 用 Trie 掩码限制只许采样以 '_tot' 开头的词"]
      G2 --> G3["模型精准采样出 [14892] ('_total') -> 剥离 '_tot' 返回 'al_cost(items):' (准确率 76.2%!)"]
    end
```

### 3.3 IDE 连续敲字下 `SPM` 前缀缓存命中与 Step-1 `Token Healing` 交互时序图

```mermaid
sequenceDiagram
    participant IDE as VS Code / Cursor IDE (`def calculate_tot|`)
    participant Heal as Token Healer (Marisa-Trie)
    participant APC as GPU Radix / Prefix Cache (SPM)
    participant Sampler as GPU Step-1 Masked Sampler

    IDE->>Heal: 1. Send `P = "...def calculate_tot"`, `S = "    return res\n..."` (2,048 tokens)
    Heal->>Heal: 2. Format `SPM`: `<|fim_prefix|><|fim_suffix|>{S}<|fim_middle|>{P}` & pop tail token `"_tot"`
    Heal->>APC: 3. Lookup `SPM` prefix -> 100% of `{S}` + 98% of `{P}` hit GPU KV Cache (`91.4%` hit rate!)
    Heal->>Sampler: 4. Query `vocab_trie.with_prefix("_tot")` (0.08 ms) -> apply bitmask to Step-1 logits
    Sampler-->>Heal: 5. Sample token `[14892]` (`"_total"`) -> strip prefix `"_tot"` -> emit `"al"` + `"_cost(items):"`!
    Heal-->>IDE: 6. Render ghost text `"al_cost(items):"` in 11.4 ms TTFT!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 代码补全 FIM 格式与分词边界方案 | 半截标识符补全准确率 | GPU 前缀缓存命中率 | P99 首字延迟 (`TTFT`) | 治愈 CPU 开销 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `PSM` 格式 (无 Token Healing)** | 前缀在前、后缀在后，直接分词 | `57.8%` (断词变傻! ✕) | `42.0%` (后缀全失效 ✕) | `48.5 ms` ✕ | `0.00 ms` |
| **Tier 2: `SPM` 格式 (无 Token Healing)** | 后缀在前缓存高，但仍被断词困扰 | `60.4%` ✕ | `91.4%` 🏆 | `11.3 ms` 🏆 | `0.00 ms` |
| **Tier 3: `PSM` + 简单退格至空格 (无 Trie 掩码)** | 删掉半截词重算，丢失已敲字母约束 | `66.5%` | `42.0%` ✕ | `48.8 ms` ✕ | `0.15 ms` |
| **Tier 4: `SPM` + 1-Token 回溯 + Trie 掩码 (推荐)** | **后缀前置缓存 + 回退末词 + 前缀掩码** | **`76.2%` (`+18.4 pt` 🏆)** | **`91.4%` 🏆** | **`11.4 ms` (`4.25x` 🏆)** | **`0.08 ms` 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_code_fim_token_healing_bpe_prefix_boundary_backtracking/run.py`](../../experiments/llm_code_fim_token_healing_bpe_prefix_boundary_backtracking_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_code_fim_token_healing_bpe_prefix_boundary_backtracking_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-code-fim-token-healing-bpe-prefix-boundary-backtracking-sizer.html`](../reports/llm-code-fim-token-healing-bpe-prefix-boundary-backtracking-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-code-fim-token-healing-bpe-prefix-boundary-backtracking-selection-cards.html`](../reports/llm-code-fim-token-healing-bpe-prefix-boundary-backtracking-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: PSM (Unhealed)", "exact_match_pct": 57.8, "ttft_ms": 48.5 },
      { "tier": "Tier 2: SPM (Unhealed)", "exact_match_pct": 60.4, "ttft_ms": 11.3 },
      { "tier": "Tier 3: PSM + Space Backtrack", "exact_match_pct": 66.5, "ttft_ms": 48.8 },
      { "tier": "Tier 4: SPM + Trie Token Healing", "exact_match_pct": 76.2, "ttft_ms": 11.4 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "exact_match_pct": "Percentage",
    "ttft_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "SPM FIM Format + Marisa-Trie Token Healing Boosts Mid-Word Code Completion Accuracy +18.4% & Cuts TTFT 4.25x",
    "subtitle": "Mid-identifier code completion exact-match accuracy (%) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "exact_match_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有基于 `Qwen2.5-Coder`、`DeepSeek-Coder`、`StarCoder2`、`CodeLlama` 的 IDE 实时代码补全服务与前缀续写接口。
- **⛔ Not for**：在连续打字补全场景下使用每次都会冲掉后缀缓存的 `PSM` 顺序，或直接把半截单词的 BPE Token 送入模型生成。

### 生产核验 Checklist
- [ ] **将 IDE 实时 FIM 请求统一构造为 `SPM` 顺序（`<|fim_prefix|><|fim_suffix|>{S}<|fim_middle|>{P}`）**：让静态后缀 `{S}` 常驻前缀缓存根部。
- [ ] **在服务启动时基于模型词表构建内存 `Marisa-Trie` 前缀索引**。
- [ ] **对末尾非空白 Token 执行 1-Token 回溯 + Step-1 前缀掩码采样 + 已敲前缀剥离**：用 `"def calculate_tot"` 与 `"https://"` 用例验证 `0` 断词失真。
