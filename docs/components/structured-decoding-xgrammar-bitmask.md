# 零开销结构化受限解码：`XGrammar` 下推自动机自适应位掩码与 vLLM V1 异步流水线

> **研究问题 ID**：`grammar-constrained-structured-decoding-xgrammar-llguidance`（`priority: 19`）
> **关联组件**：`structured-decoding-engine`
> **一手源码核验**：
> - `mlc-ai/xgrammar` (`commit 614ab01d2af3fdfdde73ae3e4171a7f36db45c23`)：
>   - `cpp/compiled_grammar_impl.h` (`blob 5a42c02fe62f69cfb36bafa0684f6708cebbd6d1`)：`AdaptiveTokenMask`（`StoreType::{kAccepted, kRejected, kAcceptedBitset}`，`USE_BITSET_THRESHOLD = 1000`，`uncertain_indices` 与 `repeat_interior_bitsets`）及 `CompiledGrammar::Impl::adaptive_token_mask_cache`。
>   - `cpp/grammar_matcher.cc` (`blob 7af9fa03cd5308bb1797adf238753d8d25b0ed1d`)：`GetBitmaskSize(vocab_size) = ceil(vocab_size / 32)` `int32` 紧凑位掩码、`TraverseDraftTreeRecursive` 投机树 `Rollback(1)` 验证与 `BatchGrammarMatcher::Impl::BatchFillNextTokenBitmask` 多线程并行填充。
> - `vllm-project/vllm` (`commit e0adacc9a67a5f2b2d8ba508f7f522bc1ff83c6d`)：
>   - `vllm/v1/structured_output/__init__.py` (`blob dc3a236aa7cc5af84bd6c7c2b133b55b138a96d6`) & `vllm/v1/structured_output/backend_xgrammar.py` (`blob aae72873b2e60147ac99301af9dd0fdd86935d2f`)：`StructuredOutputManager` 预分配 `_grammar_bitmask`、`executor_for_fillmask` 分块并行填充与 Reasoning `<think>...</think>` 延迟激活语法约束。
> **可复现离线实验**：[`experiments/structured_decoding_xgrammar/run.py`](../../experiments/structured_decoding_xgrammar/run.py)（`--check` 封印结果：[`results.json`](../../experiments/structured_decoding_xgrammar/results.json)）

---

## 1. 结构化输出在 LLM 推理中的“CPU 掩码墙”

在 Agent 工具调用（Tool Calling）、JSON Schema 强制输出、SQL/DSL 代码生成等场景中，推理引擎必须在每个 Decode 步根据当前生成前缀过滤非法 Token，将非法 Token 的 Logits 置为 $-\infty$。然而在大词表（如 Llama-3 / DeepSeek 的 `V = 128,256`）与高并发（`BS = 32..128`）下，传统动态状态机/正则匹配面临三重性能断崖：

1. **全词表动态状态扫描爆炸**：若每生成一个 Token 都在 CPU 上对 `128,256` 个候选 Token 逐字节推进下推自动机（PDA）或有限状态自动机（FSM），单请求耗时达 `0.54 ms`，在 `BS = 64`（8 核并行）下 CPU 掩码构建耗时高达 **`6.00 ms/step`**，直接把 `2.49 ms` 的 GPU 基准 Decode 步拖慢至 **`9.36 ms`**（TPOT 劣化 **`+275.32%`**）。
2. **FP32 Logit 掩码的 PCIe H2D 带宽浪费**：若每个请求传递与 Logits 同形状的 `FP32` 掩码（`128,256 × 4 B = 513,024 B ≈ 501.0 KiB/req`），`BS = 128` 每步需跨 PCIe 拷贝 **`64,128.0 KiB (62.6 MiB)`**，仅 H2D DMA 与 GPU 掩码核就白白消耗 `1.62 ms/step`。
3. **投机解码（EAGLE-3 / MTP）草稿树的状态回滚难题**：当推理引擎开启树状投机解码（每步验证 $N_{\text{tree}} = 16..64$ 个候选节点构成的树）时，语法匹配器必须沿着树拓扑一次性生成全部 $N_{\text{tree}}$ 个节点的精确位掩码，并在分支切换时以微秒级代价无损回退下推栈状态。

---

## 2. 一手源码拆解：`XGrammar` 与 `vLLM V1` 如何将受限解码压至 `< 40 us`？

### 2.1 `AdaptiveTokenMask`：预计算剔除 `97.4%` 上下文无关 Token
在 `mlc-ai/xgrammar` (`commit 614ab01`, `cpp/compiled_grammar_impl.h`, `blob 5a42c02f`, L40–116) 中，编译器针对语法图中的每个局部解析位置 `ParserState`（字符规则块内的状态），在 **离线/首次编译期** 遍历词表前缀树（Trie），将全部 `V = 128,256` 个 Token 严格划分为三类：
1. **`accepted_indices`（恒合法 Token）**：无论外层下推自动机调用栈（Pushdown Stack）处于何种历史上下文，只要当前位于该 `ParserState` 就一定合法的 Token；
2. **`rejected_indices`（恒非法 Token）**：在该 `ParserState` 的首字符/子串阶段即违反局部字符规则的 Token；
3. **`uncertain_indices`（上下文相关待定 Token，占比仅 `~2.6%`，约 `3,335` 个 Token）**：包含可跳出当前规则边界（如包含 `}`, `]`, `",` 等闭合符号）因而其合法性取决于外层调用栈顶元素的 Token。

同时，`AdaptiveTokenMask` 根据 `accepted_indices` 与 `rejected_indices` 的基数自适应选择最紧凑存储格式 `StoreType`（`USE_BITSET_THRESHOLD = 1000`）：
- 若合法集极小（`< 1000`），存为 `kAccepted` 有序下标数组；
- 若非法集极小（`< 1000`），存为 `kRejected` 有序下标数组；
- 否则预存为紧凑位图 `kAcceptedBitset`（`DynamicBitset`）。
- 针对形如 `[a-zA-Z0-9]{min,max}` 的定量重复字符类，`repeat_interior_bitsets`（`unordered_map<int, DynamicBitset>`）按剩余允许长度区间缓存全词表位集，将正则量词匹配降为 $\mathcal{O}(1)$ 查表。

在 **在线 Decode 期**（`cpp/grammar_matcher.cc`, `blob 7af9fa03`），引擎直接拷贝预计算位集并 **仅对 `~3,335` 个 `uncertain_indices` 执行运行时栈验证**，将单请求 CPU 掩码生成耗时从 `540 us` 暴降至 **`7.8 us`**（提速 **`69x`**）。

### 2.2 `int32` 紧凑位掩码（`32.0x` 压缩）与位运算内核
`XGrammar` 与 `vLLM V1` (`vllm/v1/structured_output/__init__.py`, `blob dc3a236a`, L151–165) 彻底摒弃了 `FP32` 掩码张量，改用 1-bit/token 的紧凑 `int32` 位掩码缓冲区：
$$\text{BitmaskWords}(V) = \left\lceil \frac{V}{32} \right\rceil = \left\lceil \frac{128,256}{32} \right\rceil = \mathbf{4,008\text{ words (`int32`)}} = \mathbf{16,032\text{ Bytes (15.66 KiB) / req}}$$
对比 `513,024 Bytes` 的 FP32 掩码，内存与 PCIe H2D 传输量精确压缩 **`32.0x`**（在 `BS = 64` 下从 `32,064.0 KiB` 降至 **`1,002.0 KiB`**）。GPU 侧仅需一个轻量级 Elementwise Bitwise Kernel 读取 `(logits_idx >> 5)` 对应的 `int32` 字并测试 `(word >> (logits_idx & 31)) & 1 == 0`，耗时仅 **`14–30 us`**。

### 2.3 `vLLM V1` 异步 CPU 流水线与 `TraverseDraftTreeRecursive` 树投机验证
1. **CPU-GPU 跨步异步重叠与线程池分块**：在 `vllm/v1/structured_output/__init__.py` 中，`StructuredOutputManager` 维护后台线程池 `executor_for_fillmask`（当请求数或并行度触发时按 `batch_size = 16` 分块派发至最多 `8` 个工作线程），将第 $t+1$ 步的 CPU 位掩码生成与第 $t$ 步的 GPU Transformer 前向传播流水线重叠。
2. **Reasoning `<think>...</think>` 边界感知**：`_get_constraint_bounds` 允许推理模型在 `</think>` 思维链结束标记输出前完全免除 CFG 掩码计算（零开销自由推理），仅在检测到 `</think>` 闭合后原地激活 `XGrammar` 状态机。
3. **EAGLE-3 树状投机解码的 `TraverseDraftTreeRecursive` (`grammar_matcher.cc`, L55–162)**：
   给定 `retrieve_next_token` 与 `retrieve_next_sibling` 邻接表，匹配器通过深度优先搜索（DFS）遍历整棵草稿树：每进入子节点调用 `matcher->AcceptToken(curr_token)` 并调用 `matcher->FillNextTokenBitmask` 写入该节点对应的位掩码行；从子树返回时调用 **`matcher->Rollback(1)`** 弹出下推栈历史，在 **`0.32 ms`**（`BS=32, N_tree=16`）内即可完成 `512` 行树掩码的精确构造与 H2D 传输。

---

## 3. 定量实验结果：`BS = 1..128` 吞吐与 TPOT 劣化矩阵

运行 `python experiments/structured_decoding_xgrammar/run.py --check`（词表 `V = 128,256`，上下文无关预计算覆盖率 `97.4%`，待定 Token `3,335` 个），核心结果如下：

| 并发 Batch Size | 受限解码后端 | 单步掩码体积 (KiB) | CPU 掩码耗时 | PCIe H2D 耗时 | GPU 掩码核 | 暴露结构化开销 | 端到端 TPOT | TPOT 劣化率 | 批次总吞吐 (`tok/s`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BS = 32`** | `naive_dynamic_fsm_fp32` | `16,032.0` | `3.0000 ms` | `0.3600 ms` | `0.1234 ms` | `3,483.4 us` | `5.624 ms` | `+162.70%` | `5,689.5` |
| **`BS = 32`** | `uncached_pda_fp32_sync` | `16,032.0` | `0.5897 ms` | `0.3570 ms` | `0.1070 ms` | `1,053.8 us` | `3.195 ms` | `+49.22%` | `10,016.4` |
| **`BS = 32`** | `xgrammar_bitmask_sync` | **`501.0`** | `0.0726 ms` | `0.0167 ms` | `0.0220 ms` | `111.2 us` | `2.252 ms` | `+5.20%` | `14,208.0` |
| **`BS = 32`** | **`xgrammar_async_overlap`** | **`501.0`** | `0.0709 ms` | `0.0157 ms` | `0.0220 ms` | **`37.7 us`** | **`2.179 ms`** | **`+1.76%`** | **`14,687.7`** |
| **`BS = 64`** | `naive_dynamic_fsm_fp32` | `32,064.0` | `6.0000 ms` | `0.7020 ms` | `0.1618 ms` | `6,863.8 us` | `9.357 ms` | `+275.32%` | `6,839.9` |
| **`BS = 64`** | `xgrammar_bitmask_sync` | **`1,002.0`** | `0.0726 ms` | `0.0274 ms` | `0.0300 ms` | `129.9 us` | `2.623 ms` | `+5.21%` | `24,400.2` |
| **`BS = 64`** | **`xgrammar_async_overlap`** | **`1,002.0`** | `0.0709 ms` | `0.0264 ms` | `0.0300 ms` | **`56.4 us`** | **`2.549 ms`** | **`+2.26%`** | **`25,104.2`** |
| **`BS = 128`** | **`xgrammar_async_overlap`** | **`2,004.0`** | `0.1418 ms` | `0.0478 ms` | `0.0460 ms` | **`93.8 us`** | **`3.291 ms`** | **`+2.93%`** | **`38,896.9`** |

### 核心架构结论
1. **单步暴露开销压至 `< 40 us`（`BS=32` 为 `37.7 us`，TPOT 劣化仅 `+1.76%`）**：通过 `97.4%` 词表预分类缓存 + `32x` `int32` 位掩码压缩 + 异步 CPU 重叠，结构化 JSON/CFG 输出从“吞吐杀手”变为几乎不可感知的零开销层。
2. **`BS = 64` 下实现 `3.67x` 端到端吞吐跃升**：对比朴素动态 FSM（`6,839.9 tok/s`），`xgrammar_async_overlap` 达到 **`25,104.2 tok/s`**。
3. **生产防踩坑指南**：切勿在请求进入 Decode 循环后再同步编译复杂 JSON Schema；必须在 Prefill 排队阶段通过 `StructuredOutputManager` 异步完成 `CompiledGrammar` 构建并命中全局 Schema 哈希缓存。
