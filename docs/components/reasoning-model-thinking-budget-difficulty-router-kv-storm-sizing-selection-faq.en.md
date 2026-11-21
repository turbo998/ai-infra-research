# Reasoning Model (`DeepSeek-R1 / o3 / QwQ`) Test-Time Compute, `<think>` Budget Governance, 3-Tier Difficulty Routing & `Little's Law` KV Storm Sizing Guide

> **Language / 语言**: **English (Current)** | [中文](./reasoning-model-thinking-budget-difficulty-router-kv-storm-sizing-selection-faq.md)  
> **Scope**: A production capacity planning and governance guide for Cloud & AI Solution Architects deploying reasoning models (`DeepSeek-R1`, `Qwen3-Thinking`, `o3`)—explaining **(1) why replacing a `400-token` chat model with a `3,600-token` `<think>` reasoning model triggers a `14.4x` quadratic KV Cache explosion via `Little's Law`, (2) why naive `max_tokens` truncation severs `<think>` chains before the final answer is emitted (`76.5%` accuracy), and (3) how a `65/25/10` 3-Tier Difficulty Router + dynamic `</think>` soft budget gate + multi-turn `<think>` stripping cuts active KV slots by `81.2%` and cost per task by `78.9%`**.  
> **Provenance**: Analytical sizing and routing model synthesized from published test-time compute benchmarks (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). See [`experiments/reasoning_model_thinking_budget_difficulty_router_kv_storm_sizing_selection/results.json`](../../experiments/reasoning_model_thinking_budget_difficulty_router_kv_storm_sizing_selection/results.json).

![Reasoning Model Thinking Budget Governance, Difficulty Routing & KV Storm Sizing Summary](../../assets/svg/reasoning-model-thinking-budget-difficulty-router-kv-storm-sizing-selection-faq-summary.svg)

> 🔗 **Interactive H5 Learning Lab & Companion Tools (Zero External Dependencies)**:
> - 📱 **[Interactive H5 Study Page: Reasoning `<think>` Quadratic KV Storm Math, Difficulty Router & Budget Enforcer Lab (`docs/h5/reasoning-model-thinking-budget-kv-storm-lab.html`)](../h5/reasoning-model-thinking-budget-kv-storm-lab.html)**
> - 🎓 **[AI Infra Panorama H5 Learning Hub (`docs/h5/index.html`)](../h5/index.html)**
> - 🧮 **[Interactive Thinking-Budget & KV Storm Sizer (`reasoning-model-thinking-budget-difficulty-router-kv-storm-sizing-selection-sizer.html`)](../reports/reasoning-model-thinking-budget-difficulty-router-kv-storm-sizing-selection-sizer.html)**
> - 🃏 **[7-Card 4:5 Visual Report (`reasoning-model-thinking-budget-difficulty-router-kv-storm-sizing-selection-cards.html`)](../reports/reasoning-model-thinking-budget-difficulty-router-kv-storm-sizing-selection-cards.html)**

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Under `40 RPS` enterprise traffic, sending every prompt to an unbounded `DeepSeek-R1` deployment while retaining prior `<think>` blocks in multi-turn history (Tier 1) burns an average of `3,450` thinking tokens on trivial tasks and inflates `Little's Law` active KV sessions from `320` to **`2,450`**; conversely, applying a naive `max_tokens=1024` cutoff (Tier 2) truncates complex reasoning mid-thought before `</think>`, crashing accuracy to **`76.5%`**! Deploying **Tier 4 (`<4 ms` 3-Tier Difficulty Router [`65% Fast 0-Think / 25% 1K-Think / 10% 8K-Think`] + Graceful `</think>` Budget Injection + Multi-Turn `<think>` Stripping + Native `MTP` Speculative Decoding)** preserves **`94.6%` task accuracy** while cutting average thinking tokens to **`780`**, shrinking active KV sessions by **`81.2%` (`2,450 -> 460`)**, and reducing cost per task by **`78.9%` (`$0.0180 -> $0.0038`)**!**

| Governance Tier (`40 RPS` Enterprise Mix) | Routing & Thinking-Budget Mechanism | Avg `<think>` Tokens | Peak Active KV Slots ($B_{\text{active}}$) | Task Accuracy | Cost / Task | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟥 Tier 1: Unbounded `R1` + Retained `<think>`** | Full thinking on all prompts; `<think>` kept in history | `3,450 Tok` ✕ | **`2,450 Slots` ✕ (KV Storm)** | `94.2%` | `$0.0180` ✕ | **Severe Anti-Pattern**: 72% FLOPs wasted on overthinking |
| **🟨 Tier 2: Hard `max_tokens=1024` Cutoff** | Abruptly severs output at 1,024 tokens without `</think>` | `820 Tok` | `740 Slots` | **`76.5%` ✕ (Truncated)** | `$0.0065` | **Never hard-truncate `<think>` streams** |
| **🟦 Tier 3: `65/25/10` Difficulty Router + `<think>` Strip** | Routes by complexity; strips `<think>` from multi-turn context | `960 Tok` ✅ | `820 Slots` ✅ | `94.0%` ✅ | `$0.0052` ✅ | Cuts active KV concurrency by `66.5%` |
| **🟩 Tier 4: 3-Tier Router + Graceful `</think>` Gate + `MTP` (Recommended)** | **Injects `</think>\nFinal Answer:` at budget + `MTP` 2.1x** | **`780 Tok` 🏆** | **`460 Slots` 🏆 (`-81.2%`)** | **`94.6%` 🏆** | **`$0.0038` 🏆 (`-78.9%`)** | **Production standard for Reasoning LLM fleets** |

---

## 2. 🎯 Architect Deep-Dive Q&A & Quadratic KV Storm Math

### Q1: Why does expanding output from `400` tokens to `3,600` (`<think>` + answer) multiply cluster KV HBM demand by `14.4x` at the same RPS?
$$\text{E2EL} = \text{TTFT} + (N_{\text{think}} + N_{\text{ans}} - 1)\times \text{TPOT}, \qquad B_{\text{active}} = \text{RPS} \times \text{E2EL}, \qquad M_{\text{KV\_fleet}} = B_{\text{active}} \times \left[S_{\text{in}} + 0.5(N_{\text{think}} + N_{\text{ans}})\right] \times \text{KV}_{\text{bytes/tok}}$$
- When $S_{\text{out}}$ grows `9x` (`400 -> 3,600`), `E2EL` increases **`8.44x`** (`6.88s -> 58.08s`), multiplying `Little's Law` active sessions $B_{\text{active}}$ by **`8.44x`** (`276 -> 2,324`) while average sequence length per session grows **`1.71x`** (`2,248 -> 3,848` tokens)—yielding an **`8.44 × 1.71 = 14.4x` explosion in total active KV tokens**!

### Q2: How do Graceful `</think>` Budget Enforcement and Multi-Turn `<think>` Stripping work?
1. **Separate `thinking_budget` from `answer_budget`**: When `<think>` reaches its budget (`0`, `1,024`, or `8,192` tokens), the engine injects `\n</think>\n\n` via logit forcing and reserves `512+` tokens to synthesize the final structured response.
2. **Always Strip Prior `<think>` Blocks Across Turns**: Only pass the assistant's final answer into subsequent turns so multi-turn context stays compact and prefix-cacheable.

---

## 3. 🧭 3-Tier Difficulty Router & Graceful `</think>` Gate

```mermaid
flowchart LR
    In["Incoming Request (40 RPS)"] --> Cls["3ms Difficulty Router\n(Semantic Classifier)"]
    Cls -->|"65% Simple"| L1["L1 Fast Mode (0 Think)\nImmediate <think></think> Answer"]
    Cls -->|"25% Medium"| L2["L2 Capped Think (<=1,024 Tok)\nGraceful </think> Injection"]
    Cls -->|"10% Complex"| L3["L3 Deep Think (<=8,192 Tok)\n+ Native MTP 2.1x Decode 🏆"]
```

---

## 4. 📊 Reasoning Governance Strategy Comparison Matrix

| Strategy | Simple Query Latency | Complex Task Accuracy | Multi-Turn KV Growth | Relative KV Pool Size | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Unbounded `<think>` (`Tier 1`)** | `25–60 s` ✕ | `94.2%` | Explodes (`+3.4K tok/turn`) | `100%` (OOM Risk) | Reject for production |
| **2. Hard `max_tokens` Cutoff (`Tier 2`)** | `12 s` | **`76.5%` ✕ (No Answer)** | Moderate | `30.2%` | Never hard-truncate `<think>` |
| **3. Manual `/no_think` Tagging** | `3.2 s` | `93.5%` | Clean (`<think>` stripped) | `38.0%` | Good for static APIs |
| **4. 3-Tier Auto Router + Graceful `</think>` + `MTP` (`Tier 4`)** | **`3.2 s`** 🏆 | **`94.6%`** 🏆 | **Clean (`-80%` history)** 🏆 | **`18.8%` (`-81.2%`)** 🏆 | **Enterprise Production Standard** |

---

## 5. 🛠️ Production Readiness Checklist

- [ ] **Strip `<think>...</think>` Before Appending to Multi-Turn History**: Follow the official DeepSeek-R1 contract to prevent multi-turn KV explosion.
- [ ] **Use Graceful `</think>` Injection Instead of Hard `max_tokens`**: Reserve dedicated post-`</think>` tokens so structured JSON/tool outputs never truncate mid-thought.
- [ ] **Enable Native `MTP` on Reasoning Pools**: Cutting Decode `TPOT` by `2x` directly halves `E2EL` and `Little's Law` active KV occupancy.
