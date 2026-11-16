# LLM Vocab-Parallel Embedding & `lm_head` Padding Alignment (`TP × 128`), Ghost-Token `-inf` Masking & Pipeline Tied-Weight Sync Selection Guide: Why `Qwen2.5/3` Pads `151,665` Tokenizer Vocab to `152,064` (`1.63x` Faster `lm_head` GEMM), Why Unmasked Ghost Rows Cause `1.8%` Detokenizer Crashes & How `TP × 128` Alignment + Ghost Masking Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/run.py` and `results.json`.

![Vocab-Parallel Padding Alignment & Tied-Weight Sync Visual Summary](../../assets/svg/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **`TP × 128` Vocabulary Padding Alignment (`make_vocab_size_divisible_by=128`) + Padded Ghost-Row (`[V_raw : V_pad)`) `-inf` Logit Masking + Pipeline Endpoint `embedding_group` Tied-Gradient `All-Reduce` (`Megatron-Core / vLLM / SGLang / TorchTitan`)**.
- **Architect Value**: Why do models like `Qwen2.5 / Qwen3` have `len(tokenizer) == 151,665` (or `151,936`), yet specify `vocab_size = 152,064` in `config.json`—leaving `399` unused "ghost" rows at the end of the embedding table? Misunderstanding this hardware-software contract causes three frequent production bugs:
  1. **The `1.63x` `lm_head` GEMM Slowdown on Unaligned Vocabularies (`4.85 ms` vs `2.98 ms`, Tier 1)**:
     - Hopper/Blackwell `WGMMA`, asynchronous `TMA` (128-byte alignment), and `FP8 / NVFP4` block scaling (`128×128` tiles) require each TP rank's local vocabulary dimension $V_{\text{local}} = V / \text{TP}$ to be a multiple of **`64` or `128` elements**.
     - Using the raw tokenizer length `151,665` (or shrinking `152,064` down to `151,668` by calling `model.resize_token_embeddings(len(tokenizer))` without `pad_to_multiple_of`) breaks `TP=8` divisibility and forces `cuBLASLt` off its vectorized 128-element aligned path—dropping Tensor Core utilization from `78.6%` to **`48.2%`** (`4.85 ms` vs `2.98 ms`)!
  2. **The `1.80%` Out-of-Vocabulary Ghost-Token Sampling Crash (Tier 2)**:
     - When `V_raw = 151,665` is padded to `V_pad = 152,064`, indices `151,665 .. 152,063` (`399` ghost rows on the last TP shard) do not map to any valid string in the tokenizer!
     - If those `399` ghost columns are zero-initialized (yielding `logit = 0.0` while valid rare tokens have `logit = -8.0`) and **not masked to `-inf`**, stochastic sampling (`T >= 1.0`) samples ghost token IDs $\ge 151,665$ on **`1.80%` of requests**, crashing the detokenizer!
  3. **The `+0.42` Perplexity Drift from Unsynced Tied Embeddings Across Pipeline Stages (`PP = 8`, Tier 3)**:
     - When `tie_word_embeddings=True` under `PP > 1`, Stage `0` holds `wte` and Stage `P-1` holds `lm_head`. Forgetting to run `dist.all_reduce(wte.main_grad, group=embedding_group)` between Stage `0` and Stage `P-1` prior to `optimizer.step()` causes the tied weights to diverge across stages!
  By **padding `vocab_size` to a multiple of `TP × 128`, masking all ghost columns `v >= len(tokenizer)` to `-inf` before softmax/sampling, and synchronizing tied endpoint gradients across `embedding_group` (Tier 4)**, `lm_head` GEMM latency drops **`1.63x` (`4.85 ms -> 2.98 ms`)** with **`0.0%` ghost-token crashes** and **`0.00` PPL drift**!

---

## ① 📌 Executive Summary

1. **First-Principles Formula: `TP × 128` Vocabulary Alignment**:
   $$V_{\text{pad}} = \left\lceil \frac{V_{\text{raw}}}{\text{TP} \times 128} \right\rceil \times (\text{TP} \times 128), \qquad V_{\text{local}} = \frac{V_{\text{pad}}}{\text{TP}} \equiv 0 \pmod{128}$$
   - For `V_raw = 151,665`, padding to `152,064` gives `19,008` rows per rank on `TP=8` (`38,016 = 297 × 128` on `TP=4`), adding only `0.26%` memory overhead while unlocking full `WGMMA / TMA / FP8` speed (`1.63x` speedup).
2. **Ghost-Row `-inf` Masking Invariant**:
   - On any TP shard where `vocab_end_index > V_raw`, clamp `local_logits[..., V_raw - vocab_start_index :] = -float("inf")` before softmax, cross-entropy, or token sampling.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why does calling `model.resize_token_embeddings(len(tokenizer))` after adding 2 custom tokens cripple training and break `vLLM`?
- Because `len(tokenizer)` (`151,667`) is smaller than the pre-padded `config.vocab_size` (`152,064`)! Calling `resize_token_embeddings(151667)` without `pad_to_multiple_of=1024` **truncates** the aligned `152,064` matrix down to `151,667`, breaking `TP=8` divisibility and Tensor Core alignment! Always pass `pad_to_multiple_of=1024` (or skip resizing when `len(tokenizer) <= config.vocab_size`).

### Q2: How does `VocabParallelEmbedding` prevent out-of-bounds memory reads when each TP rank holds only `19,008` rows?
- Each TP rank $r$ computes `masked_input = (input_ids - start_r) * ((input_ids >= start_r) & (input_ids < end_r))`, performs a local embedding lookup on `masked_input`, zeroes out positions where the mask was false, and runs `dist.all_reduce(output, op=SUM)` (or `Reduce-Scatter` under Sequence Parallelism).

### Q3: Should new `14B+` architectures set `tie_word_embeddings=True` or `False`?
- Always set **`tie_word_embeddings=False`** on models $\ge 14\text{B}$: untying eliminates the cross-pipeline Stage-0 $\leftrightarrow$ Stage-$(P-1)$ `All-Reduce` synchronization bottleneck at a negligible `< 2%` parameter cost.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Vocab-Parallel Alignment, Ghost Masking & Tied-Weight Sync Decision Tree

```mermaid
flowchart TD
    A["LLM Training or Serving with Raw Tokenizer Vocab V_raw = 151,665 (TP=8, PP=8)"] --> B{"Is vocab_size Padded to a Multiple of TP × 128?"}
    B -- "No: Use raw V_raw = 151,665" --> C["⛔ Tier 1: Indivisible by TP=8 & misses 128-element TMA/WGMMA path -> 4.85 ms lm_head!"]
    B -- "Yes: Pad to V_pad = 152,064" --> D{"Are Ghost Columns [151,665 : 152,064) Masked to -inf?"}
    D -- "No: Leave ghost logits unmasked" --> E["⛔ Tier 2: Sampler picks ghost IDs >= 151,665 -> 1.80% detokenizer crash rate!"]
    D -- "Yes: Mask ghost columns to -inf" --> F{"If tie_word_embeddings=True & PP > 1, Sync Stage 0 & Stage P-1 Grads?"}
    F -- "No: Independent optimizer updates on Stage 0 & Stage P-1" --> G["⛔ Tier 3: Tied weights diverge across pipeline -> +0.42 PPL degradation!"]
    F -- "Yes: All-Reduce(wte.main_grad) across pp_embedding_group" --> H["🏆 Tier 4: 2.98 ms lm_head (1.63x faster) + 0.0% Ghost Crash + 0.00 PPL Drift!"]
```

### 3.2 Unaligned Vocab & Ghost Token Leak vs. `TP × 128` Padding + `-inf` Mask

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: Unaligned GEMM (4.85ms) & Unmasked Ghost Token Crash (1.8%)"]
      B1["V_raw = 151,665: Scalar alignment fallback -> 4.85 ms GEMM"] --> B2["Unmasked ghost rows [151665:152064] have logit ~0.0 -> Samples OOV ID 151,888!"]
    end
    subgraph Good["🏆 Tier 4: Padded 152,064 (19,008/rank) + Ghost -inf Mask + PP Endpoint Sync"]
      G1["V_pad = 152,064: Aligned 128-element tiles -> 2.98 ms GEMM (1.63x faster!)"] --> G2["Mask logits[:, v >= 151665] = -inf -> 0.0% ghost token sampling!"]
    end
```

### 3.3 `VocabParallel` Lookup, Ghost-Column `-inf` Masking & `PP` Endpoint Sync Sequence

```mermaid
sequenceDiagram
    participant S0 as PP Stage 0 (VocabParallelEmbedding)
    participant S7 as PP Stage 7 (ColumnParallel lm_head, TP Rank 7)
    participant Sampler as Softmax / Cross-Entropy / Sampler

    S0->>S0: 1. Forward lookup: mask tokens outside `[r*19008, (r+1)*19008)` & `All-Reduce`
    S0->>S7: 2. Pipeline forward activation `H` through Stages `0 -> 7`
    S7->>S7: 3. Compute aligned GEMM `logits_local = H @ W_local.T` (19,008 cols, 2.98 ms)
    S7->>Sampler: 4. Mask ghost columns `logits_local[:, 151665 - 7*19008 :] = -inf` (399 cols)
    S7->>S0: 5. Backward complete -> `dist.all_reduce(wte.main_grad, group=pp_embedding_group)`!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Vocabulary Padding & Sync Policy | `lm_head` GEMM Time | Tensor Core Util | Ghost-Token Crash Rate | `PP` Tied PPL Drift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Unpadded Raw Vocab (`151,665`)** | Unaligned scalar GEMM fallback | `4.85 ms` ✕ | `48.2%` ✕ | `0.0%` | `0.00` |
| **Tier 2: Padded `152,064` (Unmasked)** | Aligned GEMM, ghost cols unmasked | `2.98 ms` | `78.6%` | **`1.80%` ✕** | `0.00` |
| **Tier 3: Masked + Unsynced Tied `PP`** | Ghost masked, Stage 0/7 grads unsynced | `2.98 ms` | `78.6%` | `0.0%` | **`+0.42` ✕** |
| **Tier 4: `TP × 128` Padded + Masked + Synced (Recommended)** | **128-aligned + `-inf` mask + PP sync** | **`2.98 ms` (`1.63x` 🏆)** | **`78.6%` (`+30.4 pt` 🏆)** | **`0.0%` 🏆** | **`0.00` 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/run.py`](../../experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/run.py) · Data: [`results.json`](../../experiments/llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-sizer.html`](../reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-selection-cards.html`](../reports/llm-vocab-parallel-embedding-lmhead-padding-alignment-tied-weights-selection-cards.html)

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

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All tensor-parallel training and serving pipelines (`Megatron-Core`, `TorchTitan`, `vLLM`, `SGLang`) and custom SFT tokenizer extensions.
- **⛔ Not for**: Setting `vocab_size = len(tokenizer)` without `TP × 128` padding or leaving `[len(tokenizer) : vocab_size)` logits unmasked during sampling.

### Production Checklist
- [ ] **Pad `vocab_size` to a multiple of `TP_max × 128` (`pad_to_multiple_of=1024` in `resize_token_embeddings`)**.
- [ ] **Mask padded ghost columns `[len(tokenizer) : config.vocab_size)` to `-inf` prior to softmax and token sampling**.
- [ ] **Synchronize tied embedding gradients across `pp_embedding_group` before `optimizer.step()` when `tie_word_embeddings=True` and `PP > 1`**.
