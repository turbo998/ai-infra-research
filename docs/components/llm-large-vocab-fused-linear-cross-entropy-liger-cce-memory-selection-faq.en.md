# Large-Vocabulary (`152K–256K Vocab`) Long-Context Training Memory Explosion & Fused Linear Cross-Entropy (`Liger-Kernel / Apple Cut-Cross-Entropy`) Selection Guide: Why `lm_head + CrossEntropyLoss` Alone Consumes `39.1 GB` at `32K` Context (`OOM`), Why Fused Softmax Still Wastes `19.8 GB` & How Zero-Logit-Materialization Fused Linear CE (`0.16–0.48 GB`, `244x` Smaller, `1.46x` Faster) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_large_vocab_fused_linear_cross_entropy_liger_cce_memory_selection/run.py` and `results.json`.

![Large-Vocab Fused Linear Cross-Entropy Visual Summary](../../assets/svg/llm-large-vocab-fused-linear-cross-entropy-liger-cce-memory-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Zero-Full-Logit-Materialization Fused Linear Cross-Entropy (`LigerFusedLinearCrossEntropyLoss` Token-Chunked In-Place Backward & `Apple Cut-Cross-Entropy (CCE)` SRAM-Tiled Online LSE + Indexed Target Dot Product) for Large-Vocabulary LLMs (`Qwen3 152K / Gemma 256K / Llama-3 128K`)**.
- **Architect Value**: When fine-tuning or RL post-training an `8B` LLM (`16 GB` weights) on an 80GB H100 at `32K` context length with FlashAttention and activation checkpointing enabled, jobs still crash with `CUDA out of memory` at the final line `loss = F.cross_entropy(lm_head(h).float(), labels)`:
  1. **Standard PyTorch Eager `lm_head + FP32 CrossEntropyLoss` (Tier 1)**: With $N = 32,768$ tokens and vocabulary $V = 152,064$, materializing `[32768, 152064]` BF16 logits (`9.97 GB`) + the upcast FP32 copy for Softmax (`19.93 GB`) + backward gradient buffers (`9.20 GB`) consumes **`39.10 GB` of HBM on the output layer alone** (`68%` of total activation memory!) and stalls steps at `1,480.0 ms`.
  2. **Fused Softmax-CE Without Fusing `lm_head` (Tier 2)**: Even with a Triton online-softmax CE kernel, executing `logits = lm_head(h)` separately still writes and stores a **`9.97 GB` BF16 logits tensor + `9.83 GB` gradient tensor (`19.80 GB` total)** in HBM.
  By **fusing the `lm_head` linear projection directly into Cross-Entropy via token-chunked in-place gradient accumulation (`Liger-Kernel`, `0.48 GB` peak, Tier 3) or SRAM-tiled online `LogSumExp` without writing logits to HBM (`Cut-Cross-Entropy`, `0.16 GB` peak, Tier 4)**, peak output-layer memory drops **`81.5x–244.4x` (`39.10 GB -> 0.16 GB`)** while eliminating `40+ GB` of HBM memory traffic to boost training throughput **`1.46x`**!

---

## ① 📌 Executive Summary

1. **Why the Output Layer Alone Uses More Memory Than All 32 Transformer Layers Combined**:
   - Modern multilingual and reasoning tokenizers (`Qwen 152,064`, `Gemma 256,000`) have a **$37.1\times \sim 62.5\times$ dimension expansion ratio** between hidden dimension $D = 4,096$ and vocabulary $V$.
   - Projecting a tiny **`0.27 GB`** hidden-state tensor $H \in \mathbb{R}^{32768 \times 4096}$ through $W_{\text{head}}^T$ inflates it $37.1\times$ into a **`10.0 GB`** BF16 tensor (and **`20.0 GB`** in FP32).
2. **First-Principles Solution (Never Materialize `[N, V]` in HBM)**:
   - Cross-entropy loss $\mathcal{L}_i = -(h_i \cdot w_{y_i}) + \log \sum_{v=1}^V \exp(h_i \cdot w_v)$ and its logit gradient $\nabla \text{logit}_{i,v} = P_{i,v} - \mathbb{I}(v = y_i)$ only ever need **one token chunk at a time (`Liger-Kernel`, $B_c = 1,024$ tokens $\to$ `0.48 GB`)** or **SRAM register tiles storing only the 1D `LSE` vector of size `[N]` (`128 KB`, `Cut-Cross-Entropy` $\to$ `0.16 GB`)**.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Does computing $\nabla H$ and $\nabla W$ inside `forward()` (`Liger-Kernel` chunked pattern) alter numerical accuracy?
- **Zero accuracy loss (100% mathematically equivalent to FP32 Cross-Entropy)**. Because scalar loss is the terminal node of the forward graph, computing unscaled unit gradients $\frac{\partial \mathcal{L}}{\partial H}$ and $\frac{\partial \mathcal{L}}{\partial W}$ chunk-by-chunk during `forward` and multiplying by scalar `grad_output` in `backward` produces identical gradients without storing `[N, V]` logits.

### Q2: What about `GRPO / PPO / DPO` where we need per-token `logprobs` and `entropy`?
- Never call `logits = model(input_ids).logits` in RL trainers! Pass `labels=action_ids` directly into a **Fused Linear Logprob + Entropy kernel** so only 1D tensors `selected_logprobs [B, S]` and `entropy [B, S]` (`128 KB` each) are returned.

### Q3: How should chunk size $B_c$ be chosen in `LigerFusedLinearCrossEntropyLoss`?
- Setting $B_c = 1,024 \sim 2,048$ tokens ensures the chunked GEMM `[B_c, D] x [D, V]` saturates GPU Tensor Cores (`> 85%` peak TFLOPS) while capping temporary buffer footprint below `0.5 GB`.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Large-Vocabulary `lm_head + Cross-Entropy` Selection Decision Tree

```mermaid
flowchart TD
    A["Training / Fine-Tuning Large-Vocab LLM (V = 128K~256K, Tokens/GPU N >= 8K)"] --> B{"How Are lm_head & Cross-Entropy Executed?"}
    B -- "Standard PyTorch: logits = lm_head(H); F.cross_entropy(logits.float(), Y)" --> C["⛔ Tier 1: Materializes 10GB BF16 + 20GB FP32 + Grad = 39.1 GB OOM!"]
    B -- "Fused Softmax-CE Only (Separate lm_head Projection)" --> D["⛔ Tier 2: Still writes & stores 19.8 GB of full [N, V] BF16 logits in HBM"]
    B -- "Fuse Linear (lm_head) + Cross-Entropy (Never Materialize [N, V])" --> E{"Tiling & Backward Gradient Strategy?"}
    E -- "Token-Chunked (B_c=1,024) In-Place Grad (Liger-Kernel)" --> F["✅ Tier 3: Peak CE memory cut 81.5x to 0.48 GB; drop-in replacement"]
    E -- "SRAM-Tiled Online LSE + Indexed Target Dot (Cut-Cross-Entropy)" --> G["🏆 Tier 4: Zero HBM logit writes! 0.16 GB peak (244.4x smaller), 1.46x faster"]
```

### 3.2 Materializing `[32K, 152K]` Logits (`39.1 GB` OOM) vs. Fused Linear CE (`0.16 GB`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: Separate lm_head + FP32 Softmax CE (39.10 GB Peak -> OOM!)"]
      B1["Hidden H [32K, 4096] (0.27 GB)"] --> B2["37x Expansion! Writes BF16 [32K, 152K] (10 GB) + FP32 Copy (20 GB) + Grad (9.1 GB)"]
    end
    subgraph Good["🏆 Tier 3/4: Fused Linear Cross-Entropy (0.16 – 0.48 GB Peak)"]
      G1["Hidden H [32K, 4096] (0.27 GB)"] --> G2["1,024-Token Chunk or SRAM-Tiled Online LSE -> Never Materializes Full [N, V] Logits!"]
    end
```

### 3.3 Fused Linear Cross-Entropy Forward-Backward Execution Sequence

```mermaid
sequenceDiagram
    participant HBM as GPU HBM (Hidden H & Weight W)
    participant SRAM as GPU L2 / SRAM (1,024-Token Tile)
    participant Kernel as Fused Linear + Online Softmax Kernel

    HBM->>SRAM: 1. Stream chunk H_c [1024, 4096] & vocabulary weight tiles W
    SRAM->>Kernel: 2. Compute tile logits H_c @ W^T & online Softmax (LSE_c, P_c - Y_c)
    Kernel->>HBM: 3. Accumulate dH_c = (P_c - Y_c) @ W and dW += (P_c - Y_c)^T @ H_c in-place
    Note over SRAM,Kernel: 4. Discard tile buffer immediately -> 0.16-0.48 GB peak memory!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | `lm_head + Cross-Entropy` Architecture | Peak Logits/CE Memory (`32K`) | Step Latency | Training Throughput | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Standard Eager (`BF16 + FP32 CE`)** | Full `BF16` + `FP32` `[N, V]` logits | `39.10 GB` (OOM! ✕) | `1,480.0 ms` ✕ | `22,140 tok/s` ✕ | ⛔ **Guaranteed OOM at Long Context** |
| **Tier 2: Fused `Softmax-CE` Only** | Separate `lm_head` writes `[N, V]` | `19.80 GB` ✕ | `1,190.0 ms` ✕ | `27,536 tok/s` ✕ | ⛔ **20 GB Unnecessary HBM Bloat** |
| **Tier 3: Chunked Fused Linear CE (`Liger`)** | `1,024`-token chunk in-place grad | `0.48 GB` (`81.5x` Lower ✅) | `1,045.0 ms` ✅ | `31,356 tok/s` ✅ | ✅ **Drop-In SFT / RL Integration** |
| **Tier 4: SRAM-Tiled Online LSE (`Cut-CE`) (Recommended)** | **Indexed target dot + SRAM LSE** | **`0.16 GB` (`244.4x` Lower 🏆)** | **`1,012.0 ms` 🏆** | **`32,379 tok/s` (`1.46x` 🏆)** | 🏆 **Long-Context & Ultra-Large Vocab** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_large_vocab_fused_linear_cross_entropy_liger_cce_memory_selection/run.py`](../../experiments/llm_large_vocab_fused_linear_cross_entropy_liger_cce_memory_selection/run.py) · Data: [`results.json`](../../experiments/llm_large_vocab_fused_linear_cross_entropy_liger_cce_memory_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-large-vocab-fused-linear-cross-entropy-liger-cce-memory-sizer.html`](../reports/llm-large-vocab-fused-linear-cross-entropy-liger-cce-memory-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-large-vocab-fused-linear-cross-entropy-liger-cce-memory-selection-cards.html`](../reports/llm-large-vocab-fused-linear-cross-entropy-liger-cce-memory-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Standard Eager BF16+FP32 CE", "mem_gb": 39.10, "step_ms": 1480.0 },
      { "tier": "Tier 2: Fused Softmax-CE Only", "mem_gb": 19.80, "step_ms": 1190.0 },
      { "tier": "Tier 3: Liger Chunked Fused Linear CE", "mem_gb": 0.48, "step_ms": 1045.0 },
      { "tier": "Tier 4: Cut-Cross-Entropy SRAM-Tiled LSE", "mem_gb": 0.16, "step_ms": 1012.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "mem_gb": "Quantity",
    "step_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Fused Linear Cross-Entropy (Liger / Cut-CE) Cuts 152K-Vocab Final-Layer Memory from 39.1 GB to 0.16 GB",
    "subtitle": "Peak lm_head + Cross-Entropy HBM footprint (GB) at 32K tokens/GPU (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "mem_gb",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All SFT, continual pretraining, DPO, and GRPO workloads on models with $V \ge 64\text{K}$ (`Qwen 152K`, `Gemma 256K`, `Llama-3 128K`).
- **⛔ Not for**: Materializing `logits = self.lm_head(hidden_states)` before calling `nn.CrossEntropyLoss()`.

### Production Checklist
- [ ] **Bypass full `[B, S, V]` logit materialization in `CausalLM.forward()`**: Pass `(hidden_states, lm_head.weight, labels)` directly into `LigerFusedLinearCrossEntropyLoss` or `linear_cross_entropy` (CCE).
- [ ] **Use fused linear logprob + entropy kernels in RLHF/GRPO**: Return only `[B, S]` token logprobs and entropy scalars.
- [ ] **Verify chunk size (`1,024–2,048` tokens)**: Maintain high Tensor Core occupancy while keeping temporary buffers `< 0.5 GB`.
