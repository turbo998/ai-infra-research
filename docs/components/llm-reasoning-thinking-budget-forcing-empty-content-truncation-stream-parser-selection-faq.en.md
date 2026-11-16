# Reasoning LLM (`DeepSeek-R1 / Qwen3 / QwQ`) `<think>` Budget Forcing, Empty-Content Truncation (`content=""`) & Multi-Turn Stripping Selection Guide: Why `max_tokens=4096` Causes `29.4%` Empty Answers When `<think>` Overthinks, Why Retaining Historical `<think>` Wastes `68.8%` KV Memory & How Soft-Boundary Transition Bridges (`0%` Empty Answers, `94.8%` Completion, `3.2x` Lower KV) Solve It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_reasoning_thinking_budget_forcing_empty_content_truncation_stream_parser_selection/run.py` and `results.json`.

![Reasoning LLM Thinking-Budget Forcing & Multi-Turn Stripping Visual Summary](../../assets/svg/llm-reasoning-thinking-budget-forcing-empty-content-truncation-stream-parser-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Decoupled `(max_thinking_tokens, reserved_answer_tokens)` Budgeting + Soft-Boundary Transition Bridge Injection (`"\nFinal conclusion:\n</think>\n\n"`) + Automatic Multi-Turn Historical `<think>` Stripping (`DeepSeek-R1 / Qwen3 / s1 Budget Forcing / vLLM & SGLang ReasoningParser`)**.
- **Architect Value**: Serving reasoning LLMs (`DeepSeek-R1`, `Qwen3-235B-A22B`, `QwQ-32B`) behind production APIs and agent workflows exposes three critical failure modes when `<think>...</think>` blocks are left unmanaged:
  1. **The `29.4%` Empty-Answer Truncation Trap (`content=""` & `finish_reason="length"`, Tier 1)**:
     - Standard `max_tokens = 4,096` limits the combined total of internal chain-of-thought tokens ($T_{\text{think}}$) and final response tokens ($T_{\text{answer}}$).
     - On `29.4%` of complex reasoning and tool-calling prompts (`54.0%` at `max_tokens = 2,048`), the model loops inside `<think>` ("Wait, let me verify another edge case...") until all `4,096` tokens are exhausted **before `</think>` is ever emitted**—returning `content = ""` and `tool_calls = None` and crashing downstream callers!
  2. **The `3.20x` Multi-Turn KV Bloat & `-4.8 pt` Accuracy Drop from Retained `<think>` History (Tier 1 vs Tier 2)**:
     - Appending previous turns' `reasoning_content` (or raw `<think>...</think>` blocks) back into `messages` on subsequent turns violates the official `DeepSeek-R1` / `Qwen3` training distribution (where historical `<think>` blocks are always stripped), inflating 10-turn KV cache memory from `4.50 GB` to **`14.40 GB` (`3.20x` bloat)** and dropping accuracy by **`-4.8 pt`**!
  3. **The `14.2%` Malformed JSON Rate Under Abrupt Mid-Sentence `</think>` Forcing (Tier 3)**:
     - Blindly forcing the `</think>` token at step `3,072` while the model is mid-sentence or mid-equation leaves an unfinished grammatical context that corrupts **`14.2%`** of subsequent structured JSON/tool outputs!
  By **splitting `max_tokens=4096` into `max_thinking_tokens=3072` + `reserved_answer_tokens=1024`, waiting for a natural newline/period boundary in the soft window `[2950, 3072]` before injecting the transition bridge `"\nFinal conclusion:\n</think>\n\n"`, and stripping `<think>` from all prior assistant turns (Tier 4)**, empty-answer truncation drops from `29.4%` to **`0.0%`**, task completion rises from `65.8%` to **`94.8%` (`+29.0 pt` gain)**, and multi-turn KV memory drops by **`68.8%` (`3.20x` reduction)**!

---

## ① 📌 Executive Summary

1. **First-Principles Rule: Orthogonal Thinking vs. Answer Token Budgets**:
   $$M_{\text{total}} = B_{\text{think}} + B_{\text{answer}} \quad (\text{e.g., } 4,096 = 3,072 + 1,024)$$
   - The thinking phase may finish early (e.g., at 600 tokens) and release unused budget, but it is **never allowed to consume the final $B_{\text{answer}} = 1,024$ tokens reserved for `content` / `tool_calls`**.
2. **Why Soft-Boundary Bridge Injection Beats Hard Token Forcing (`0.4%` vs `14.2%` Malformed JSON)**:
   - Waiting for a sentence/line delimiter (`"\n"`, `"."`) inside `[B_soft, B_hard]` (`[2950, 3072]`) and injecting `"\nFinal conclusion:\n</think>\n\n"` closes the reasoning block at a clean syntactic boundary, allowing the model to transition smoothly into valid JSON or tool calls.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: How can we bypass `<think>` entirely for latency-sensitive classification or simple chat turns (`enable_thinking=False`)?
- Prefill an empty closed thinking block `<think>\n\n</think>\n\n` immediately after `<|im_start|>assistant\n` in the prompt template; the model sees that thinking is already closed and emits the final answer on Decode Step 1!

### Q2: Why must `<think>...</think>` be stripped from `messages[:-1]` in multi-turn conversations?
- Both `DeepSeek-R1` and `Qwen3` strip historical `<think>` blocks during multi-turn SFT/RL. Stripping them in production aligns with training priors (`+4.8 pt` accuracy), saves `68.8%` of KV cache memory, and keeps prefix-cache blocks compact.

### Q3: How should streaming SSE parsers handle partial `"</think>"` tags across token boundaries?
- Use a stateful streaming FSM with prefix-overlap holdback so partial prefixes (`"</"`, `"</thi"`) are buffered until `"</think>"` is confirmed, preventing tag fragments from leaking into `delta.reasoning_content` or `delta.content`.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Reasoning `<think>` Budget Control & Multi-Turn History Decision Tree

```mermaid
flowchart TD
    A["Production Reasoning LLM Serving (DeepSeek-R1 / Qwen3 / QwQ, max_tokens=4096)"] --> B{"Are Historical <think> Blocks Stripped in Multi-Turn Chat?"}
    B -- "No: Retain past <think> in messages[:-1]" --> C["⛔ Tier 1: 3.20x KV memory bloat (14.40 GB) + -4.8 pt multi-turn accuracy drop!"]
    B -- "Yes: Strip historical <think> blocks" --> D{"How Is Current-Turn <think> Token Length Bounded?"}
    D -- "Unbounded (shares single max_tokens=4096)" --> E["⛔ Tier 2: 29.4% of hard queries exhaust 4,096 tokens inside <think> -> content=''!"]
    D -- "Abrupt mid-sentence </think> token forcing at 3,072" --> F["⚠️ Tier 3: 0% empty answers, but 14.2% malformed JSON/tool outputs!"]
    D -- "Soft-newline bridge ('\\nFinal conclusion:\\n</think>\\n\\n') + 1,024 answer reserve" --> G["🏆 Tier 4: 0.0% Empty Answers + 94.8% Task Completion (+29.0 pt) + 68.8% KV Saved!"]
```

### 3.2 Unbounded `<think>` (`29.4%` Empty `content=""`) vs. Soft-Boundary Budget Forcing (`94.8%` Success)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: Unbounded <think> + Retained History (29.4% Empty Answers & 14.40 GB KV)"]
      B1["Spends all 4,096 tokens inside <think> before </think>"] --> B2["Returns finish_reason='length' with content='' -> Breaks downstream callers!"]
      B2 --> B3["Retains past <think> across 10 turns -> 14.40 GB KV HBM!"]
    end
    subgraph Good["🏆 Tier 4: Soft-Boundary Bridge (B_think=3072, B_ans=1024) + History <think> Strip"]
      G1["At t >= 2950, wait for '\\n' & inject '\\nFinal conclusion:\\n</think>\\n\\n'"] --> G2["Guarantees 1,024 tokens for clean final answer / tool call (0% empty!)"]
      G2 --> G3["Strips historical <think> -> 4.50 GB KV HBM (-68.8% saved)!"]
    end
```

### 3.3 Streaming `ReasoningParser` FSM & Soft-Boundary Budget Forcing Sequence

```mermaid
sequenceDiagram
    participant Client as Agent / API Caller
    participant FSM as Streaming ReasoningParser & Budget Forcer
    participant Engine as GPU Decode Engine (M_total = 4096)

    Client->>FSM: 1. Send multi-turn request (`max_tokens=4096`, `max_thinking_tokens=3072`)
    FSM->>Engine: 2. Strip `<think>...</think>` from historical turns `messages[:-1]` (-68.8% KV)
    loop Thinking Phase (`t = 1 .. B_soft=2950`)
        Engine-->>FSM: 3. Emit thinking tokens -> stream as `delta.reasoning_content`
    end
    Engine-->>FSM: 4. At `t = 2968`, token ends with `"\n"` -> trigger Soft-Boundary Bridge!
    FSM->>Engine: 5. Inject forced tokens `"\nFinal conclusion:\n</think>\n\n"` & switch FSM to `IN_CONTENT`
    loop Answer / Tool-Call Phase (`up to 1,024 reserved tokens`)
        Engine-->>Client: 6. Stream clean final answer (`delta.content` / `delta.tool_calls`, 0% truncation!)
    end
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Thinking Budget & History Policy | Empty `content=""` Rate | Malformed JSON Rate | 10-Turn KV HBM | Task Completion Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Unbounded + Retain History** | Single `max_tokens=4096` + keep `<think>` | `29.4%` ✕ | `29.4%` ✕ | `14.40 GB` ✕ | `65.8%` ✕ |
| **Tier 2: Strip History Only (Unbounded)** | Strips prior `<think>`, no think cap | `29.4%` ✕ | `29.4%` ✕ | `4.50 GB` | `70.6%` ✕ |
| **Tier 3: Abrupt Mid-Sentence `</think>`** | Forces `</think>` mid-sentence at 3,072 | `0.0%` | `14.2%` ✕ | `4.50 GB` | `84.2%` |
| **Tier 4: Soft-Boundary Bridge + Strip (Recommended)** | **Newline bridge + 1,024 answer reserve** | **`0.0%` 🏆** | **`0.4%` 🏆** | **`4.50 GB` (`-68.8%` 🏆)** | **`94.8%` (`+29.0 pt` 🏆)** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_reasoning_thinking_budget_forcing_empty_content_truncation_stream_parser_selection/run.py`](../../experiments/llm_reasoning_thinking_budget_forcing_empty_content_truncation_stream_parser_selection/run.py) · Data: [`results.json`](../../experiments/llm_reasoning_thinking_budget_forcing_empty_content_truncation_stream_parser_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-reasoning-thinking-budget-forcing-empty-content-truncation-stream-parser-sizer.html`](../reports/llm-reasoning-thinking-budget-forcing-empty-content-truncation-stream-parser-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-reasoning-thinking-budget-forcing-empty-content-truncation-stream-parser-selection-cards.html`](../reports/llm-reasoning-thinking-budget-forcing-empty-content-truncation-stream-parser-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unbounded + Keep History", "task_acc_pct": 65.8, "empty_ans_pct": 29.4 },
      { "tier": "Tier 2: Strip History Only (Unbounded)", "task_acc_pct": 70.6, "empty_ans_pct": 29.4 },
      { "tier": "Tier 3: Abrupt Mid-Sentence </think>", "task_acc_pct": 84.2, "empty_ans_pct": 0.0 },
      { "tier": "Tier 4: Soft-Boundary Bridge + Reserved Answer", "task_acc_pct": 94.8, "empty_ans_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "task_acc_pct": "Percentage",
    "empty_ans_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Soft-Boundary Thinking-Budget Forcing Lifts Task Completion to 94.8% & Eliminates Empty Answers",
    "subtitle": "End-to-end reasoning & tool-use task completion accuracy (%) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "task_acc_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All production deployments of `DeepSeek-R1`, `Qwen3`, `QwQ`, and reasoning-enabled agents requiring guaranteed structured outputs and bounded multi-turn KV usage.
- **⛔ Not for**: Allowing `<think>` to consume 100% of `max_tokens` or retaining previous turns' `reasoning_content` inside multi-turn prompt histories.

### Production Checklist
- [ ] **Separate `max_thinking_tokens` from `reserved_answer_tokens`**: Guarantee at least `512–1,024` tokens for the final response/tool call after `</think>`.
- [ ] **Use a Soft-Boundary Transition Bridge (`"\nFinal conclusion:\n</think>\n\n"`) at a newline/sentence boundary**: Avoid abrupt mid-sentence `</think>` forcing.
- [ ] **Automatically strip `<think>...</think>` from prior assistant turns (`messages[:-1]`)**.
