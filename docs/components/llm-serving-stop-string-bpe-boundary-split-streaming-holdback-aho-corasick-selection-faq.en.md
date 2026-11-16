# LLM Serving (`vLLM / SGLang / TGI`) Stop-String BPE Token-Boundary Split, Streaming Holdback & Incremental `Aho-Corasick` Selection Guide: Why `tokenizer.encode(stop)` Misses `31.4%` of Stops When BPE Merges `":\n"`, Why Full-String `.find()` Leaks Partial Stops to SSE & Burns `18.40 ms` CPU, and How Incremental `Aho-Corasick` + Dynamic Prefix-Overlap Holdback (`0.12 ms/step`, `153.3x` Faster, `0.0%` Leak) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_serving_stop_string_bpe_boundary_split_streaming_holdback_aho_corasick_selection/run.py` and `results.json`.

![LLM Serving Stop-String BPE Split & Incremental Aho-Corasick Visual Summary](../../assets/svg/llm-serving-stop-string-bpe-boundary-split-streaming-holdback-aho-corasick-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Incremental Byte/Char `Aho-Corasick` Automaton + Dynamic Prefix-Overlap Holdback Buffer (`p = ac_dfa.depth(state)`) + Sub-Token Tail Slicing (`vLLM V1 StopChecker / SGLang Incremental Detokenizer`)**.
- **Architect Value**: In ReAct agent loops (`stop=["\nObservation:", "</tool_call>", "\n```\n"]`), naive stop-string handling causes two production-breaking bugs:
  1. **The BPE Token-Boundary Split Trap (`31.4%` Missed Stops under Token-ID Matching, Tier 1)**:
     - Standalone `tokenizer.encode("\nObservation:")` yields `[198 ("\n"), 35012 ("Observation"), 25 (":")]`.
     - However, when the LLM generates `"\nObservation:\n"` in context, the BPE tokenizer merges the trailing colon and newline into a **single distinct token `[510] (":\n")`**! Similarly, `".\nObservation:"` merges `".\n"` into `[624]`.
     - Because token ID `25` (`":"`) **never appears in the generated token stream**, engines that only check `stop_token_ids` miss **`31.4%`** of valid stop points, causing the LLM to hallucinate fake tool outputs and dropping ReAct loop completion to **`64.2%`**!
  2. **The Full-String `O(B * k * L)` Scan & Streaming Prefix Leak Trap (`18.40 ms/step` & `100%` SSE Leak, Tier 2)**:
     - Scanning the entire generated text (`L = 4,096` chars) across `B = 256` requests and `k = 8` stop strings on every decode step burns **`18.40 ms/step`** of Python CPU time.
     - Worse, when `stream=True` and `include_stop_str_in_output=False`, flushing each token immediately sends `"\nObserv"` over SSE before `"ation:"` is generated on the next step—leaking partial stop strings to **`100.0%`** of streaming responses!
  By **advancing a per-request `Aho-Corasick` DFA only on newly decoded token characters (`0.12 ms/step`, `153.3x` faster), dynamically holding back only the $p = \text{depth}(\text{dfa\_state})$ bytes that overlap a valid stop-string prefix (`0` lag when $p = 0$), and slicing sub-token tails (`":\n"` $\to$ `""`) (Tier 4)**, missed stops and SSE prefix leaks both drop to **`0.0%`** and ReAct agent loop completion reaches **`99.6%` (`+35.4 pt`)**!

---

## ① 📌 Executive Summary

1. **First-Principles Rule: Multi-Character Stop Strings Must Be Matched on the Incrementally Detokenized Character Stream, Never Solely via `stop_token_ids`**:
   - Only atomic `special_tokens` (e.g., `<|eot_id|>`, `<|im_end|>`) are guaranteed never to merge with adjacent characters in BPE. Ordinary stop strings can merge at both their leading edge (`".\n"`) and trailing edge (`":\n"`, `">\n"`), requiring character-level suffix matching and **sub-token tail truncation**.
2. **Dynamic Prefix-Overlap Holdback Law (`p = ac_automaton.depth(state)`)**:
   - In an Aho-Corasick trie built over the request's stop strings $\mathcal{S}$, the depth $p$ of the current DFA state after feeding `pending_buf + new_token_str` equals the **exact length of the longest suffix of the buffer that is a prefix of any stop string in $\mathcal{S}$**.
   - Therefore, on every step we immediately flush `candidate[:-p]` to SSE (`100%` of the token when $p = 0$) and retain only the last $p$ bytes in `pending_buf`!

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why is incremental `Aho-Corasick` `153.3x` faster than Python `full_text.find()` at `L = 4,096`?
- `full_text.find()` rescans all `4,096` characters $\times 8$ stop strings $= 32,768$ character comparisons per request on every token (`8.38M` comparisons across `B = 256` requests = `18.40 ms`). Incremental `Aho-Corasick` transitions only on the newly decoded token (~`4` characters per request, independent of $L$ and $k$), taking **`0.12 ms/step`**.

### Q2: How does Dynamic Prefix-Overlap Holdback (Tier 4) beat Fixed Max-Length Holdback (Tier 3)?
- Fixed holdback blindly withholds $\max_{s \in \mathcal{S}} |s|$ characters on every step—even when the text has zero overlap with any stop string—causing visible streaming lag when a long stop string is configured. Dynamic holdback withholds $p = 0$ bytes whenever the current tail does not match any stop-string prefix.

### Q3: How should incomplete multi-byte UTF-8 sequences (Byte-Fallback BPE tokens) be handled before stop matching?
- Buffer incomplete UTF-8 leading bytes inside the **Incremental Detokenizer** first; only pass valid decoded UTF-8 codepoints into the stop-string DFA so invalid `U+FFFD` replacement characters never corrupt stop matching or leak to SSE.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Stop-String Matching & Streaming Holdback Decision Tree

```mermaid
flowchart TD
    A["Streaming LLM Request with stop=['\\nObservation:', '</tool_call>']"] --> B{"Stop Target Type?"}
    B -- "Atomic Special Token (<|eot_id|>)" --> C["✅ Match Directly via GPU/CPU stop_token_ids"]
    B -- "Multi-Character Text Stop Strings" --> D{"Matching & Streaming Buffer Policy?"}
    D -- "Token-ID Match Only (tokenizer.encode(stop))" --> E["⛔ Tier 1: BPE merges ':\\n' into token 510 -> Misses 31.4% of stops!"]
    D -- "Full-String .find() + Immediate SSE Flush" --> F["⛔ Tier 2: 18.40 ms/step CPU stall + 100% partial stop leak to SSE!"]
    D -- "Fixed Max-Length Blind Holdback" --> G["⚠️ Tier 3: Constant streaming lag of M_max chars, 1.45 ms/step"]
    D -- "Incremental Aho-Corasick + Dynamic Prefix Holdback p" --> H["🏆 Tier 4: 0.12 ms/step (153.3x faster) + 0.0% Miss + 0.0% Leak!"]
```

### 3.2 BPE Boundary Split (`31.4%` Miss) vs. Incremental `Aho-Corasick` + Dynamic Holdback

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: Token-ID Miss (31.4%) & Unbuffered SSE Leak (100%)"]
      B1["Generated IDs: [198 ('\\n'), 35012 ('Observation'), 510 (':\\n')]"] --> B2["Token-ID expects [198, 35012, 25 (':')] -> 510 != 25 Misses Stop!"]
      B2 --> B3["Unbuffered SSE flushes '\\nObserv' before 'ation:' arrives -> Client JSON breaks!"]
    end
    subgraph Good["🏆 Tier 4: Incremental Aho-Corasick + Dynamic Prefix Holdback (p = depth)"]
      G1["Tokens '\\n', 'Observation' -> DFA depth p=12 -> Hold back 12 bytes from SSE"] --> G2["Token 510 (':\\n') -> DFA hits terminal match! Slice out ':\\n' & stop cleanly!"]
    end
```

### 3.3 Streaming Decode Step with Incremental Detokenizer & Aho-Corasick Holdback Sequence

```mermaid
sequenceDiagram
    participant Sampler as GPU Sampler
    participant Detok as Incremental UTF-8 Detokenizer
    participant AC as Per-Request Aho-Corasick DFA (pending_buf)
    participant SSE as Client SSE Stream

    Sampler->>Detok: 1. Token t=1: ID 35012 -> decoded delta `"Observation"`
    Detok->>AC: 2. Feed `"Observation"` after `"\n"` -> DFA depth p = 12 (prefix of `"\nObservation:"`)
    AC->>AC: 3. Retain `p = 12` bytes in `pending_buf = "\nObservation"` (emit `""` to SSE)
    Sampler->>Detok: 4. Token t=2: ID 510 -> decoded delta `":\n"`
    Detok->>AC: 5. Feed `":"` -> DFA hits terminal match for `"\nObservation:"`!
    AC->>SSE: 6. Truncate at `":"`, discard held-back stop string & send `finish_reason="stop"`
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Stop-String Matching & Buffer Policy | BPE Split Missed Stops | Streaming SSE Prefix Leak | CPU Match Latency (B=256) | ReAct Agent Loop Success |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Token-ID Stop Only** | Matches `tokenizer.encode(stop)` IDs | `31.4%` ✕ | `0.0%` | `0.08 ms` | `64.2%` ✕ |
| **Tier 2: Full-String `.find()` (No Holdback)** | Scans full `output_text` every step | `0.0%` | `100.0%` ✕ | `18.40 ms` ✕ | `72.5%` ✕ |
| **Tier 3: Fixed Max-Length Blind Holdback** | Always withholds last $M_{\max}$ chars | `0.0%` | `0.0%` | `1.45 ms` | `94.0%` |
| **Tier 4: Incremental Aho-Corasick + Overlap (Recommended)** | **Advances DFA on delta + holds back $p$** | **`0.0%` 🏆** | **`0.0%` 🏆** | **`0.12 ms` (`153.3x` 🏆)** | **`99.6%` (`+35.4 pt` 🏆)** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_serving_stop_string_bpe_boundary_split_streaming_holdback_aho_corasick_selection/run.py`](../../experiments/llm_serving_stop_string_bpe_boundary_split_streaming_holdback_aho_corasick_selection/run.py) · Data: [`results.json`](../../experiments/llm_serving_stop_string_bpe_boundary_split_streaming_holdback_aho_corasick_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-serving-stop-string-bpe-boundary-split-streaming-holdback-aho-corasick-sizer.html`](../reports/llm-serving-stop-string-bpe-boundary-split-streaming-holdback-aho-corasick-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-serving-stop-string-bpe-boundary-split-streaming-holdback-aho-corasick-selection-cards.html`](../reports/llm-serving-stop-string-bpe-boundary-split-streaming-holdback-aho-corasick-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Token-ID Stop Only", "cpu_ms": 0.08, "agent_success_pct": 64.2 },
      { "tier": "Tier 2: Full-String .find() (No Holdback)", "cpu_ms": 18.40, "agent_success_pct": 72.5 },
      { "tier": "Tier 3: Blind Max-Length Holdback", "cpu_ms": 1.45, "agent_success_pct": 94.0 },
      { "tier": "Tier 4: Incremental Aho-Corasick + Overlap Holdback", "cpu_ms": 0.12, "agent_success_pct": 99.6 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "cpu_ms": "Quantity",
    "agent_success_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Incremental Aho-Corasick + Dynamic Holdback Achieves 99.6% Agent Loop Success at 0.12 ms CPU Overhead",
    "subtitle": "ReAct agent loop success rate (%) across 4 stop-string matching tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "agent_success_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All OpenAI-compatible streaming servers, ReAct/tool-use agent loops, and code-completion endpoints using multi-character `stop` strings.
- **⛔ Not for**: Relying solely on `tokenizer.encode(stop)` token IDs or scanning full output strings without prefix-overlap holdback.

### Production Checklist
- [ ] **Test BPE boundary merges (`".\nObservation:"`, `"\nObservation:\n"`, `"</tool_call>\n"`)**: Verify that stop strings trigger even when leading or trailing characters merge into adjacent BPE tokens.
- [ ] **Maintain a per-request `Aho-Corasick` DFA state and `pending_holdback` buffer**: Hold back only the $p = \text{depth}(\text{state})$ overlapping prefix bytes from SSE.
- [ ] **Slice sub-token tails on stop hits**: When `":\n"` completes `"\nObservation:"`, strip both the stop string (when `include_stop_str_in_output=False`) and the trailing `"\n"`.
