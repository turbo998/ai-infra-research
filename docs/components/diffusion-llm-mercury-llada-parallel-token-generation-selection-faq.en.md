# Discrete Diffusion LLMs (`Mercury` / `LLaDA` / `Block Diffusion BD3-LM`) Parallel Token Generation vs. Autoregressive Decoding Selection Guide (`Standard AR Decode` vs. `AR + EAGLE-3 Speculative` vs. `Full-Sequence Masked Diffusion LLaDA/MDLM` vs. `Block Diffusion + Inter-Block KV Cache + Confidence Early-Exit`)

> **Reader's Guide (3-Minute Executive Scan / 10-Minute Whiteboard Pitch)**: Written for **AI Solution Architects and LLM Serving / AI Coding Platform Leads**. Using first principles, this guide explains the 2025–2026 paradigm shift in ultra-fast token generation—**why standard Autoregressive (`AR`) decoding (Tier 1) must read the entire 8B/70B model weights from HBM `512 sequential times` to generate 512 tokens, leaving `98.2%` of an H100's Tensor Cores idle at `112.0 tok/s`; why naive full-sequence masked diffusion LLMs (`Vanilla LLaDA / MDLM`, Tier 3) cannot cache the prompt in a KV Cache due to full bidirectional attention, slowing down to `195.0 tok/s` at 4K context; and how Semi-Autoregressive Block Diffusion (`BD3-LM` / `Inception Mercury Coder`, Tier 4) combines inter-block causal KV caching with intra-block 128-token parallel Tensor Core denoising and confidence early-exit to generate 512 tokens in just `28 weight reads` at `1,180.0 tok/s` (`10.54x` faster than AR, `4.14x` faster than `EAGLE-3`) while cutting generation cost by `-86.5%`.**

[![Back to Main Catalog](https://img.shields.io/badge/⬅️_Back_to_AI_Infra_Decision_Catalog-0F172A?style=for-the-badge)](../../README.en.md)
[![中文版](https://img.shields.io/badge/🇨🇳_中文版-diffusion--llm--mercury--llada--parallel--token--generation--selection--faq.md-2563EB?style=for-the-badge)](./diffusion-llm-mercury-llada-parallel-token-generation-selection-faq.md)
[![7-Card Benchmark Report](https://img.shields.io/badge/📊_7--Card_Visual_Report-Open_HTML-9333EA?style=for-the-badge)](../reports/diffusion-llm-mercury-llada-parallel-token-generation-selection-cards.html)
[![Interactive Sizer](https://img.shields.io/badge/🧮_Interactive_Diffusion_LLM_Sizer-Open_HTML-059669?style=for-the-badge)](../reports/llm-inference-engine-pd-sizer.en.html)

![Discrete Diffusion LLM (Mercury / LLaDA / Block Diffusion BD3-LM) vs Autoregressive Selection Summary](../../assets/svg/diffusion-llm-mercury-llada-parallel-token-generation-selection-faq-summary.svg)

---

## ① 📌 Executive Summary

1. **Why Autoregressive (`AR`) Decoding (Tier 1) Hits the HBM Memory-Bandwidth Wall (`1.8%` Tensor Core Utilization)**:
   - Because standard autoregressive models generate strictly 1 token per forward pass ($P(x_t \mid x_{<t})$), generating a 512-token code completion requires **512 sequential HBM weight reads**. At low batch sizes, each weight read performs a memory-bound vector-matrix multiply (`GEMV`), utilizing only **`1.8%` of an H100's Tensor Cores ✕** and capping 8B speed at **`112.0 tok/s` (`$0.89 / 1M` tokens ✕)**.
2. **When to Use `AR + EAGLE-3 Speculative Decoding` (Tier 2) vs. Why `Vanilla Full-Sequence LLaDA / MDLM` (Tier 3) Fails on Long Prompts**:
   - **Use Tier 2 (`AR + EAGLE-3`) for drop-in acceleration of existing AR weights**: Verifying a 5-token draft tree in 1 pass accepts `~2.8–3.2` tokens/step, cutting weight reads to **`180`** and lifting speed to **`285.0 tok/s` ✅**—but remains fundamentally bottlenecked by left-to-right sequential dependencies.
   - **Why Tier 3 (`Naive Full-Sequence Masked Diffusion — Vanilla LLaDA / MDLM`) Collapses on `>=1K` Prompts**: While discrete masked diffusion unmasks 512 tokens in just `32` parallel steps, applying full bidirectional attention across the entire `4,096-token prompt + 512-token output` **prevents KV caching of the prompt**! Recomputing the 4,096-token prompt 32 times drags effective throughput down to **`195.0 tok/s` ✕**.
3. **The `>1,000 tok/s` Breakthrough Architecture (Tier 4: Semi-Autoregressive `Block Diffusion BD3-LM / Mercury Coder` + Inter-Block KV Cache + Confidence Early-Exit)**:
   - **Three First-Principles Innovations**:
     1. **Inter-Block Causal KV Cache**: Prefills the 4,096-token prompt **once** into a standard causal KV Cache and generates output in `B = 128-token` blocks. Completed blocks are appended directly to the KV Cache!
     2. **Intra-Block Parallel Bidirectional Denoising (`128-Token Tile` on Tensor Cores)**: Inside the active 128-token block, all 128 `[MASK]` tokens attend bidirectionally to each other and causally to the cached prompt—turning memory-bound `GEMV` into compute-dense **`128×128` Tensor Core `GEMM` (`42.5%` Tensor Core utilization 🏆)**!
     3. **Confidence-Threshold Early Exit ($p > 0.96$)**: Boilerplate code syntax and JSON brackets converge in `5–7` steps per block, requiring only **`28 total HBM weight reads` for 512 tokens**!
   - **Quantified Outcome**: Reaches **`1,180.0 tok/s` on 1×H100 (`10.54x` faster than AR, `4.14x` faster than `EAGLE-3` 🏆)**, achieves **`84.4%` code generation/infilling accuracy 🏆**, and slashes cost by **`-86.5%` (`$0.12 / 1M` tokens 🏆)**.

---

## ② 🎯 Architect & Customer First-Principles Q&A

### Q1: How does a Discrete Diffusion LLM ("unmasking tokens") work mathematically on discrete vocabulary IDs?

- **Iterative Parallel Unmasking (Solving a Crossword Puzzle Instead of Typing Left-to-Right)**:
  - Instead of adding Gaussian noise to continuous pixels, discrete masked diffusion (`MDLM / LLaDA`) starts with $B=128$ `[MASK]` tokens. In each denoising step, a bidirectional Transformer predicts probability distributions for all 128 positions simultaneously and locks in (`unmasks`) the top $k \approx 16\text{–}20$ highest-confidence tokens (such as `def`, `return`, parentheses, and known identifiers) first, using them as bidirectional anchors to resolve the remaining interior tokens in 5–7 steps!

### Q2: How does `Block Diffusion (BD3-LM / Mercury)` (Tier 4) enable KV Caching while keeping parallel bidirectional generation?

- **Hybrid Block-Causal Attention Mask**:
  - By enforcing a **block-causal mask** (tokens in Block $b$ can attend to all tokens within Block $b$ bidirectionally, and to all tokens in Prompt + Blocks $1 \dots b-1$ causally), the keys and values of the 4,096-token prompt and all prior blocks become **100% invariant** once committed—unlocking full Paged KV Cache reuse with `128-token` parallel Tensor Core steps!

### Q3: Where should architects deploy `Block Diffusion LLMs` (Tier 4) vs. keeping `Autoregressive + EAGLE-3` (Tier 2)?

- **Workload Selection Boundary**:
  1. **Deploy Tier 4 (`Block Diffusion / Mercury Coder`)** for **IDE real-time code completion, full-file diff editing (`Cursor Tab / Instant Apply`), structured JSON generation, and sub-500ms voice/UI agents** where `>1,000 tok/s` and bidirectional infilling are game-changing.
  2. **Keep Tier 2 (`AR + EAGLE-3`)** when serving existing pre-trained autoregressive weights (`DeepSeek-R1`, `Llama-3.3-70B`) without diffusion retraining or running 32K+ token sequential reasoning chains.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree / Roofline Comparison / Sequence)

### 3.1 Token Generation Paradigm Selection Decision Tree

```mermaid
flowchart TD
    Start["⚡ LLM Generation Paradigm Selection<br/>Inputs: Latency Target (tok/s), Prompt Length & Model Weights"] --> Q1{"Must you serve existing unmodified<br/>autoregressive checkpoints (e.g. stock Llama/DeepSeek)?"}
    Q1 -- "Yes (Zero retraining allowed)" --> Q2{"Need interactive speed (>200 tok/s)?"}
    Q2 -- "No (Offline batch)" --> T1["⚪ Tier 1: Standard Autoregressive (AR) Decode<br/>512 weight reads for 512 tokens (112 tok/s, 1.8% Tensor Core util)"]
    Q2 -- "Yes (Interactive serving)" --> T2["🔵 Tier 2: Autoregressive + EAGLE-3 Speculative Decoding<br/>✅ 2.54x lossless speedup (285 tok/s, 180 weight reads)"]
    Q1 -- "No (Can deploy Diffusion LLM: Mercury / BD3-LM / LLaDA)" --> Q3{"Does the workload have prompts >= 1,024 tokens<br/>requiring KV Cache reuse?"}
    Q3 -- "No (<256-token unconditional short text)" --> T3["🟡 Tier 3: Full-Sequence Masked Diffusion (Vanilla LLaDA / MDLM)<br/>⚠️ No KV cache! Recomputes 4K prompt 32 times (drops to 195 tok/s)"]
    Q3 -- "Yes (IDE Code Completion / Edit, JSON, Real-Time Agents)" --> T4["🟢 Tier 4: Block Diffusion (BD3-LM / Mercury) + Inter-Block KV Cache<br/>+ Confidence-Threshold Early Exit<br/>🏆 28 weight reads | 1,180 tok/s (10.54x faster) | $0.12/1M (-86.5%)"]
```

### 3.2 Autoregressive `512` Serial Weight Reads vs. Tier 4 Block Diffusion `28` Parallel Tensor Core Passes

```mermaid
flowchart LR
    subgraph AR["❌ Tier 1/2: Autoregressive Serial Decode (HBM Bandwidth Bound)"]
        direction TB
        A1["Generate 512 Code Tokens<br/>Strictly 1 token (or ~3 draft tokens) per step"] --> A2["512 Sequential HBM Weight Reads (180 with EAGLE-3)<br/>💾 Streams 8B weights from HBM 512 times"]
        A2 --> A3["1.8% - 5.4% Tensor Core Utilization<br/>💥 Capped at 112 - 285 tok/s (4.57s for 512 tokens)"]
    end

    subgraph BD["✅ Tier 4: Block Diffusion BD3-LM / Mercury (Compute-Dense Tensor Core Parallelism)"]
        direction TB
        B1["① Inter-Block Causal KV Cache<br/>4,096-token prompt prefilled ONCE into Paged KV Cache"] --> B2["② Intra-Block 128-Token Bidirectional Denoising<br/>⚡ Updates all 128 positions per weight read on Tensor Cores (42.5% util)"]
        B2 --> B3["③ Confidence Early-Exit (p > 0.96) + Warm-Start Code Edit<br/>🏆 Only 28 weight reads for 512 tokens | 1,180 tok/s (0.43s) | $0.12/1M"]
    end
```

### 3.3 End-to-End Sequence of `Block Diffusion` Generating 512 Tokens Across 4 Cached Blocks

```mermaid
sequenceDiagram
    autonumber
    participant IDE as IDE Client (4,096-Token Repo Context)
    participant KV as Inter-Block Causal KV Cache (HBM)
    participant TC as H100 Tensor Cores (128-Token Diffusion Engine)

    Note over IDE,KV: Stage 1: Single-Pass Prompt Prefill into Causal KV Cache
    IDE->>KV: Send 4,096-token file context -> prefill once and lock into KV Cache

    Note over KV,TC: Stage 2: Semi-Autoregressive Block Denoising (4 Blocks × ~7 Steps = 28 Total Passes)
    loop For Block b = 1..4 (Each Block = 128 [MASK] Positions)
        TC->>TC: Steps 1..2: 128-token GEMM pass unmasks high-confidence syntax anchors (~40 tokens)
        TC->>TC: Steps 3..6: Bidirectional attention fills interior logic & identifiers (~22 tokens/step)
        TC->>TC: Step 7 (Early Exit): All 128 positions exceed confidence p > 0.96 -> commit block early!
        TC->>KV: Append Block b's K/V tensors to Causal KV Cache for Block b+1
    end
    TC-->>IDE: Return 512-token code completion in 0.43s (1,180.0 tok/s)!
```

---

## ④ 📊 Architect Decision Matrix

| Evaluation Dimension | Tier 1: Standard `AR` Decode | Tier 2: `AR + EAGLE-3` Speculative | Tier 3: Full-Sequence Diffusion (`LLaDA`) | Tier 4: `Block Diffusion` (`BD3-LM / Mercury`) |
| :--- | :--- | :--- | :--- | :--- |
| **Core Generation & Cache Mechanism** | Causal mask, 1 token/step | 5-token draft tree + 1-pass verify | Full bidirectional, 32 steps (**No KV Cache**) | **Inter-block KV Cache + intra-block 128-tok diffusion** |
| **Generation Throughput (`tok/s` on H100)** | `112.0 tok/s` ✕ (`4.57s` / 512 tok) | **`285.0 tok/s` ✅ (`2.54x` faster)** | `195.0 tok/s` ✕ (Recomputes 4K prompt) | **`1,180.0 tok/s` 🏆 (`10.54x` faster, `0.43s` / 512 tok)** |
| **HBM Weight Reads per 512 Tokens** | `512 reads` ✕ | `180 reads` ✅ | `32 reads` (over 4,608-len sequence) | **`28 reads` 🏆 (over 128-len active block, `18.3x` fewer)** |
| **GPU Tensor Core Utilization (`%`)** | **`1.8%` ✕ (98.2% memory stall)** | `5.4%` ⚠️ | `68.0%` (Wasted on prompt recompute) | **`42.5%` 🏆 (True compute-dense Tensor Core GEMM)** |
| **Code & Infilling Accuracy (`%`)** | `83.5%` (Requires FIM hack) | `83.5%` | `82.8%` | **`84.4%` 🏆 (Native bidirectional infilling)** |
| **Cost per 1M Generated Tokens (`USD`)** | `$0.89 / 1M` ✕ | `$0.35 / 1M` ✅ | `$0.51 / 1M` ✕ | **`$0.12 / 1M` 🏆 (`-86.5%` cost reduction)** |
| **Best Fit & Architect Verdict** | Legacy baseline | **Best drop-in speedup for existing AR models** | Short `<256-tok` unconditional text only | **Gold standard for >1,000 tok/s coding, JSON & agents** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card Benchmark Insights

> **Provenance Disclosure**: All metrics in this guide and the companion 7-Card report are generated by [`experiments/diffusion_llm_mercury_llada_parallel_decoding_selection/run.py`](../../experiments/diffusion_llm_mercury_llada_parallel_decoding_selection/run.py) and sealed in [`results.json`](../../experiments/diffusion_llm_mercury_llada_parallel_decoding_selection/results.json) (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`), calibrated against Inception Labs `Mercury Coder`, `Block Diffusion (BD3-LMs)`, and `LLaDA / MDLM` benchmarks.

- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/diffusion-llm-mercury-llada-parallel-token-generation-selection-cards.html`](../reports/diffusion-llm-mercury-llada-parallel-token-generation-selection-cards.html)
- **Interactive Diffusion LLM vs. Autoregressive Sizer**: [`docs/reports/llm-inference-engine-pd-sizer.en.html`](../reports/llm-inference-engine-pd-sizer.en.html)

---

## ⑥ 🛠️ Appendix: Production Readiness Checklist

- [ ] **1. Never deploy KV-cacheless full-sequence diffusion (`Vanilla LLaDA/MDLM`) on long prompts (`>=1,024 tokens`)**: Always use Semi-Autoregressive Block Diffusion (`BD3-LM / Mercury`) so the prompt is prefilled once into a causal KV Cache.
- [ ] **2. Align active diffusion block size ($B = 64$ or $128$) with GPU Tensor Core MMA tile dimensions**.
- [ ] **3. Enable Confidence-Threshold Early Exit ($\min_i p_{\max}(x_i) > 0.96$)** to commit structured code/JSON blocks in 5–7 steps instead of fixed 16–32 steps.
- [ ] **4. Use Warm-Start Denoising for IDE code edits (`Apply Diff`)**: Initialize the block with existing code tokens to finish full-function edits in 2–4 steps.
- [ ] **5. Use `EAGLE-3` + Piecewise CUDA Graphs (Tier 2) when serving unmodified autoregressive model checkpoints**.
- [ ] **6. Enable Low-Confidence Remasking between denoising steps** to prevent parallel token collisions on adjacent high-entropy positions.
