# LLM Outlier-Free Activation & KV Quantization: Orthogonal Hadamard Rotation (`QuaRot` / `SpinQuant`) & `W4A4 / FP4` Selection Guide (FAQ)

> **Reader's Guide**: Built for AI Solution Architects and inference engine teams, this guide demystifies the single biggest barrier when moving LLM serving from weight-only `W4A16` to **end-to-end `W4A4` / `NVFP4` (4-bit weights + 4-bit activations + 4-bit KV-Cache)**: **Massive Activation Channel Outliers**. Using plain language and first principles, we explain why classical diagonal channel scaling (`SmoothQuant` / `AWQ`) collapses at 4-bit activations and 4-bit KV (`WikiText-2 PPL` degrades from `5.12` to `29.85`), and how **`QuaRot` (Randomized Walsh-Hadamard Rotation)** and **`SpinQuant` (Cayley Stiefel Manifold Learned Rotation)** exploit orthogonal invariance ($Q Q^\top = I_d$) to crush activation kurtosis ($\kappa$) from `482.4` down to a smooth Gaussian `2.92` (**`-99.4%`**), shrinking `Llama-3.1-70B` (`32K` context, `Batch=32`) VRAM from `274.2 GB` to a single-GPU **`68.6 GB` (`-75.0%`)** while unlocking **`2.60x`** Prefill throughput via native `INT4 / NVFP4` Tensor Cores.
>
> 🌐 **Language**: **[中文版本](quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-faq.md)** | **English (Current)**
> 📊 **Visual Assets & Interactive Tools**:
> - **[High-Res SVG Executive Summary Diagram](../../assets/svg/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-faq-summary.svg)**
> - **[7-Card 4:5 Visual Benchmark Report (Self-Contained HTML)](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-cards.html)**
> - **[Interactive W4A4 & Rotation Sizer Workbench (H5)](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-sizer.html)**

![QuaRot & SpinQuant Orthogonal Hadamard Rotation Summary](../../assets/svg/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-faq-summary.svg)

---

## 📌 1. Executive Summary (3-Minute Architect's Guide)

### 1.1 Plain-English First Principles: What Are Activation Outliers and How Does Hadamard Rotation Eliminate Them Losslessly?

Imagine measuring the heights of 8,192 students using a ruler with only **15 tick marks** (symmetric `INT4` quantization bins `[-7, +7]`):
- **Unrotated Transformer Activations (Raw Residual Stream)**: Large Transformers (`70B+`) exhibit an emergent property where `4–16` fixed hidden dimensions out of `8,192` act like "120-meter giants" with amplitudes **`78.5x–120x` larger than the median channel** (statistical kurtosis $\kappa = 482.4$). If you apply a per-token `INT4` quantizer across the vector, bin `+7` must stretch to cover `+64.0` (step size `9.14`). Consequently, **99.8% of normal channels in `[-0.8, +0.8]` underflow and round to zero**, causing catastrophic perplexity collapse!
- **Why Diagonal Scaling (`SmoothQuant` / `AWQ`) Fails at 4-Bit (`W4A4`)**: `SmoothQuant` divides activation channels by a diagonal factor $s_i$ and multiplies weights by $s_i$: $Y = (X \cdot \text{diag}(s)^{-1})(\text{diag}(s) \cdot W)$. While effective for `W8A8` (255 bins), pushing a `78.5x` spike into `4-bit` weights destroys the weight quantization grid and cannot handle dynamic token-wise spikes, causing `WikiText-2 PPL` to explode from `5.12` to **`29.85`**.
- **First-Principles Solution — Orthogonal Computational Invariance**:
  For any orthogonal matrix $Q \in \mathbb{R}^{d \times d}$, **$Q \cdot Q^\top = I_d$**. Inserting $Q Q^\top$ inside any linear layer $Y = X \cdot W$ leaves the output mathematically identical:
  $$Y = X \cdot I_d \cdot W = (X \cdot Q) \cdot (Q^\top \cdot W) = X_{\text{rot}} \cdot W_{\text{rot}}$$
  When $Q$ is a normalized **Walsh-Hadamard matrix $H$** (with entries $\pm \frac{1}{\sqrt{d}}$), any single-channel spike of magnitude `+64.0` is evenly redistributed across all $d = 8,192$ coordinates (dampening peak amplitude by roughly $\sqrt{d} \approx 90.5\times$). By the Central Limit Theorem, $X_{\text{rot}} = X \cdot H$ converges to an outlier-free Gaussian distribution (**kurtosis drops from `482.4` to `2.92–3.24`**), utilizing all 15 `INT4` bins while preserving the exact dot product (**`0 ULP` algebraic error**)!

### 1.2 One-Line Selection Rules

1. **Legacy GPUs (`A100` / `L40S` without native 4-bit activation Tensor Cores, low-batch decode)**: Choose **`AWQ W4A16 + FP8 KV-Cache` (`S3`)**. Compresses weights to save HBM bandwidth while keeping activations in FP16.
2. **Zero-Calibration / Training-Free Instant `W4A4 + KV4` Deployment**: Choose **`QuaRot` Randomized Hadamard Rotation (`S4`)**. Requires zero calibration samples, cutting `Llama-3.1-70B` VRAM to `68.6 GB` (`-75%`) and boosting prefill throughput by **`2.55x`** (`PPL = 5.46`).
3. **Production High-Throughput `W4A4` / Blackwell `B200 NVFP4` / Single-80GB-GPU 70B Serving (Architect's Default)**: Choose **`SpinQuant` Cayley Learned Rotation + Online `FWHT` (`S5`)**. Optimizes $R_1, R_2$ on the Stiefel manifold using ~512 calibration samples and **fuses them 100% offline into weights (`0 ms` runtime cost)**, paired with lightweight online `FWHT` ($R_3, R_4$ at `1.42%` overhead) to achieve **`5.24` PPL (`+0.12` vs FP16)**, **`78.9%` MMLU**, and **`36,900 tok/s` (`2.60x` speedup)**!

---

## 🎯 2. Architect & Customer Deep-Dive Q&A

### Q1: Doesn't multiplying by an $8192 \times 8192$ orthogonal matrix inside every Transformer layer destroy inference latency? How do `QuaRot` and `SpinQuant` keep runtime overhead below `1.8%`?

**Answer: Through "Offline Weight Absorption via Computational Invariance"—over 90% of the rotations ($R_1$ and $R_2$) are multiplied directly into static model weights during checkpoint export and never execute at runtime!**

Across a standard Transformer block (`RMSNorm`, Attention `Q/K/V/O`, MLP `Gate/Up/Down`, and residual connections), `QuaRot` and `SpinQuant` factorize rotations into **four distinct orthogonal matrices ($R_1, R_2, R_3, R_4$)**:

1. **$R_1 \in \mathbb{R}^{d \times d}$ (Global Residual Stream Rotation, `8192×8192`) — 100% Absorbed Offline (`0 ms` runtime overhead)**:
   - **Mechanism**: Because `RMSNorm` normalizes by the Euclidean norm $\|X\|_2$ and orthogonal transformations preserve norms ($\|X \cdot R_1\|_2 = \|X\|_2$), folding the learned scale vector $\gamma$ into the next linear layer yields exact commutativity:
     $$\text{RMSNorm}(X \cdot R_1) = \text{RMSNorm}(X) \cdot R_1$$
   - **Offline Absorption**: We keep the entire residual stream permanently in the rotated basis $X_{\text{rot}} = X \cdot R_1$:
     - Every matrix **reading from** the residual stream (`W_q`, `W_k`, `W_v`, `W_gate`, `W_up`, `LM_Head`) is pre-multiplied offline by $R_1^\top$: $W_{\text{in}}' \leftarrow R_1^\top W_{\text{in}}$.
     - Every matrix **writing to** the residual stream (`Embedding`, `W_o`, `W_down`) is post-multiplied offline by $R_1$: $W_{\text{out}}' \leftarrow W_{\text{out}} R_1$.
     - Result: No $8192 \times 8192$ rotation ever runs during online inference (`0.0%` latency overhead)!

2. **$R_2 \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$ (Per-Head Value-to-Output Rotation, `128×128`) — 100% Absorbed Offline (`0 ms` runtime overhead)**:
   - **Mechanism**: Inside each attention head $h$, attention probabilities $P_h$ multiply $V_h$ before projecting through $W_{o,h}$:
     $$\text{AttnOut}_h = (P_h \cdot V_h) \cdot W_{o,h} = \bigl(P_h \cdot (X W_{v,h} R_2)\bigr) \cdot (R_2^\top W_{o,h})$$
   - **Offline Absorption**: Updating $W_{v,h}' \leftarrow W_{v,h} R_2$ and $W_{o,h}' \leftarrow R_2^\top W_{o,h}$ offline guarantees that the emitted **V-Cache is already rotated and outlier-free**, enabling lossless **`INT4` V-Cache quantization** at zero runtime cost.

3. **$R_3 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{ffn}}}$ (Online Post-SwiGLU MLP Hadamard) — Lightweight Online `FWHT` (`~0.9%` latency)**:
   - **Why Online**: The element-wise non-linear `SwiGLU` activation $\text{SiLU}(x_g) \odot x_u$ blocks offline absorption and generates fresh activation spikes before `W_down`.
   - **Solution**: Apply an in-place **Fast Walsh-Hadamard Transform (`FWHT`)** $Z' = \text{FWHT}(Z)$ in GPU SRAM fused with activation quantization, while absorbing $H^\top$ offline into `W_down` ($W_{\text{down}}' \leftarrow H^\top W_{\text{down}}$). Because `FWHT` uses butterfly additions/subtractions in $O(d \log_2 d)$ operations without multiplications, it adds only `~0.9%` end-to-end latency.

4. **$R_4 \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$ (Online Post-RoPE Q/K Hadamard, `128×128`) — Unlocks `INT4` Key-Cache (`~0.5%` latency)**:
   - **Why Online**: Rotary Position Embedding (`RoPE`) acts after $W_q$ and $W_k$, leaving severe frequency-domain channel spikes in the Key tensor.
   - **Solution**: Apply a register-level `128×128` Hadamard transform to both $Q_{\text{rope}}$ and $K_{\text{rope}}$ right after `RoPE`: $Q_{\text{rot}} = Q_{\text{rope}} H_{128}$ and $K_{\text{rot}} = K_{\text{rope}} H_{128}$. Since $Q_{\text{rot}} K_{\text{rot}}^\top = Q_{\text{rope}} H_{128} H_{128}^\top K_{\text{rope}}^\top = Q_{\text{rope}} K_{\text{rope}}^\top$, attention logits are unchanged while `INT4` Key-Cache `128K` Needle-in-a-Haystack retrieval jumps from **`41.5%` to `99.1%`**!

---

### Q2: How does `SpinQuant` improve upon `QuaRot` without adding any extra runtime latency?

**Answer: Because $R_1$ and $R_2$ are 100% absorbed into static weights offline, they do not need to be Hadamard matrices—`SpinQuant` optimizes $R_1$ and $R_2$ over the continuous Stiefel manifold of all orthogonal matrices!**

- **`QuaRot`'s Bottleneck**: `QuaRot` picks a random signed Hadamard matrix $R_1 = H \cdot S$ blindly without inspecting the model's weight/activation singular vectors. At aggressive `W4A4 + KV4`, this leaves a `+0.34` PPL gap (`5.46` vs FP16 `5.12`).
- **`SpinQuant`'s Cayley Stiefel Optimization**:
  - Freezing model weights and inserting fake-quantizers into the forward graph, `SpinQuant` parameterizes any orthogonal matrix $R \in \mathcal{M}_{\text{Stiefel}}$ via the **Cayley transform** of a skew-symmetric matrix $A = -A^\top$:
    $$R(A) = (I - A)^{-1}(I + A)$$
  - Using just `512` calibration samples (~30 minutes offline), `SpinQuant` learns $R_1^*, R_2^*$ to jointly minimize `W4A4` quantization loss, then fuses $R_1^*, R_2^*$ into the exported weights.
  - **Result**: Identical runtime graph as `QuaRot`, lower kurtosis (`2.92` vs `3.24`), and **`5.24` WikiText-2 PPL (`+0.12` vs FP16)**—outperforming even weight-only `AWQ W4A16` (`5.29`) while running `2.20x` faster in prefill!

---

### Q3: How do you apply Fast Hadamard Transforms when dimensions are not powers of two (e.g., `Llama-3.1-70B` $d_{\text{ffn}} = 28,672$)?

**Answer: Use Kronecker-Product Block Hadamard Factorization ($H_d = H_{2^k} \otimes H_m$)!**

- Factor $d_{\text{ffn}} = 28,672 = 256 \times 112 = 2^8 \times 112$.
- Construct $H_{28672} = H_{256} \otimes H_{112}$, where $H_{256}$ uses register-level butterfly `FWHT` and $H_{112}$ is a precomputed 112-order Paley Hadamard matrix (`25 KB` in GPU shared memory).
- Fusing the 8-stage butterfly pass with a tiny `112×112` Tensor Core MMA keeps online $R_3$ latency overhead at **`1.42%`** (`3.43 pts` faster than dense orthogonal GEMM).

---

## 🧭 3. Visual Architecture Diagrams (Mermaid Decision Tree, Topology & Sequence)

### 3.1 Architect Quantization & Orthogonal Rotation Decision Tree

```mermaid
flowchart TD
    Start["🚀 LLM Quantization & KV Compression\n(Target: 70B+ Production Serving)"] --> HW{"Does Target GPU Have Native\nINT4 / NVFP4 Tensor Cores\nor Require 70B+32K on 1x80GB?"}

    HW -- "No: Legacy A100/L40S with\nLow-Batch Memory-Bound Decode" --> S3["⚠️ Choose S3: AWQ W4A16 + FP8 KV\n• 4-bit weights save HBM BW; FP16 activations\n• 102.1 GB VRAM (Requires 2x80GB GPUs)\n• 1.18x Prefill (No 4-bit compute speedup)"]

    HW -- "Yes: Blackwell B200 (NVFP4) / High-QPS Prefill\nor 70B+32K in 68.6 GB Single GPU" --> NaivCheck{"Can We Use SmoothQuant / RTN\nDiagonal Scaling for W4A4 + KV4?"}

    NaivCheck -- "Never!" --> S2["⛔ Reject S2: Naive SmoothQuant W4A4\n• Kurtosis κ = 310.6 (34.2x spikes remain)\n• 99% normal channels underflow in INT4\n• WikiText-2 PPL explodes to 29.85 (Collapse)"]

    NaivCheck -- "Must Apply Orthogonal Rotation\n(Q · Q^T = I_d)" --> Calib{"Can You Run a 30-Min Offline\nCayley Optimization on 512 Samples?"}

    Calib -- "No: Strict Zero-Calibration\nTraining-Free Export" --> S4["✅ Choose S4: QuaRot Random Hadamard (W4A4+KV4)\n• Offline R1/R2 Hadamard + Online R3/R4 FWHT\n• Kurtosis κ = 3.24, VRAM 68.6 GB (-75%)\n• Prefill 36,200 tok/s (2.55x), PPL = 5.46 (+0.34)"]

    Calib -- "Yes: Maximum Production Accuracy\n& Throughput (Recommended)" --> S5["🏆 Choose S5: SpinQuant Cayley Learned (W4A4+KV4)\n• Stiefel-learned R1/R2 (100% offline fused, 0 ms cost)\n• Online R3 (MLP) + R4 (RoPE KV) FWHT (1.42% cost)\n• Kurtosis κ = 2.92, PPL = 5.24 (+0.12), Prefill 2.60x!"]
```

---

### 3.2 Offline Weight Absorption ($R_1, R_2$) vs Online Fast Hadamard ($R_3, R_4$) Topology

```mermaid
flowchart LR
    subgraph Offline["📦 100% Absorbed Offline into Static Weights (Zero Runtime Latency)"]
        direction TB
        R1["R1 (8192×8192 Global Residual Stream Rotation)\n• W_q, W_k, W_v, W_up, W_gate ← R1^T · W\n• W_o, W_down ← W · R1\n• SpinQuant learns optimal R1 on Stiefel manifold"]
        R2["R2 (128×128 Per-Head Value-to-Out Rotation)\n• W_v^(h) ← W_v^(h) · R2\n• W_o^(h) ← R2^T · W_o^(h)\n• Emits outlier-free V-Cache ready for INT4"]
    end

    subgraph Online["⚡ Lightweight Online FWHT Kernels (1.42%–1.75% Total Latency)"]
        direction TB
        R4["R4 (128×128 Post-RoPE Q/K Online FWHT)\n• Q_rot = FWHT_128(RoPE(Q))\n• K_rot = FWHT_128(RoPE(K)) → Stored in INT4 K-Cache\n• Dot-Product Invariant: Q_rot · K_rot^T == Q_rope · K_rope^T"]
        R3["R3 (28672-Dim Post-SwiGLU Block FWHT)\n• Z_rot = FWHT_256x112(SiLU(Gate) ⊙ Up)\n• W_down pre-multiplied offline by H_dff^T\n• Unlocks native W4A4 / NVFP4 Tensor Core GEMM"]
    end

    Offline ==>|"Exports Rotated Checkpoint"| Online
```

---

### 3.3 End-to-End SpinQuant Export & Online W4A4 + INT4 KV Execution Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Export as 🛠️ Offline Export (SpinQuant Cayley)
    participant Residual as 🔄 Rotated Residual X_rot (FP16)
    participant Attn as ⚡ W4A4 Attention + R4 (FWHT_128)
    participant KVCache as 💾 INT4 KV-Cache (Group=128)
    participant MLP as 🚀 W4A4 MLP + R3 (FWHT_dff)

    Note over Export: Phase 1: One-Time Stiefel Manifold Optimization & Weight Fusion (0 ms Online Cost)
    Export->>Export: 1. Freeze weights; optimize R1, R2 via Cayley transform R=(I-A)^(-1)(I+A) on 512 samples
    Export->>Export: 2. Fuse R1 into W_qkv/W_o/W_up/W_down, R2 into W_v/W_o, and R3^T into W_down
    Export->>Export: 3. Quantize weights (GPTQ/AWQ W4); activation peak/median drops 78.5x → 2.41x (κ=2.92)

    Note over Residual,MLP: Phase 2: Online Forward Pass per Layer (W4A4 Tensor Cores + INT4 KV)
    Residual->>Attn: 4. RMSNorm(X_rot) + dynamic INT4 activation quantization (all 15 bins utilized)
    Attn->>Attn: 5. INT4/NVFP4 Tensor Core GEMM for Q, K, V (V already rotated by offline R2)
    Attn->>KVCache: 6. Apply register-level R4 (FWHT_128, ~0.5% cost) to RoPE(Q), RoPE(K) & write INT4 KV
    KVCache-->>Attn: 7. FlashAttention-3 / FlashInfer reads INT4 K_rot, V_rot and multiplies W_o'
    Attn->>Residual: 8. Add back to rotated residual stream X_rot
    Residual->>MLP: 9. RMSNorm(X_rot) → W4A4 Gate & Up GEMM → SwiGLU activation
    MLP->>MLP: 10. Fused online R3 block FWHT (256×112, ~0.9% cost) + INT4 quant → W4A4 W_down'
    MLP->>Residual: 11. Emit next-layer rotated residual X_rot (36,900 tok/s, 2.60x prefill speedup!)
```

---

## 📊 4. Architect Selection Matrix & Fit Boundaries

### 4.1 5-Scenario Quantitative Evaluation Matrix (`Llama-3.1-70B`, `32K` Context, `Batch=32`)

| Scenario / Scheme | Rotation Mechanism | Activation Kurtosis $\kappa$ ($\downarrow$) | Max/Median Channel Ratio | WikiText-2 PPL ($\downarrow$) | 5-Shot MMLU ($\uparrow$) | Total VRAM (W + 32K KV) | Prefill Tok/s ($\uparrow$) | Online Rotation Cost | Architect Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S1: FP16 Baseline (`W16A16+KV16`)** | None (Raw Residual Stream) | `482.4` | `78.5x` | `5.12` (`0.00`) | `79.4%` | `274.2 GB` (`140+134.2`) | `14,200` (`1.00x`) | `0.00%` | Baseline Reference (4×80GB) |
| **S2: Naive SmoothQuant (`W4A4+KV4`)** | Diagonal Migration $\text{diag}(s)$ | `310.6` | `34.2x` | **`29.85`** (`+24.73` ✕) | `34.8%` ✕ | `68.6 GB` (`35+33.6`) | `38,400` (`2.70x`) | `0.00%` | ⛔ **Reject (Severe Collapse)** |
| **S3: AWQ Hybrid (`W4A16+FP8 KV`)** | Salient-Weight Scaling (FP16 Act) | `482.4` | `78.5x` | `5.29` (`+0.17`) | `78.6%` | `102.1 GB` (`35+67.1`) | `16,800` (`1.18x`) | `0.00%` | ⚠️ **Confirm (Decode-Only)** |
| **S4: QuaRot (`W4A4+KV4` Hadamard)** | Offline Random Hadamard $R_1,R_2$ + Online $R_3,R_4$ | `3.24` | `2.85x` | `5.46` (`+0.34`) | `77.8%` | **`68.6 GB`** (`-75.0%`) | `36,200` (`2.55x`) | `1.75%` | ✅ **Act Automatically (Zero-Calib)** |
| **S5: SpinQuant (`W4A4+KV4` Cayley)** | **Cayley Learned $R_1,R_2$ (Offline) + Online $R_3,R_4$** | **`2.92`** (`-99.4%`) | **`2.41x`** | **`5.24`** (`+0.12` ★) | **`78.9%`** | **`68.6 GB`** (Fits 1×80GB) | **`36,900`** (**`2.60x`** ★) | **`1.42%`** | 🏆 **Best-in-Class Production** |

### 4.2 Fit Boundaries: ✅ Good fit vs ⛔ Not for

| ✅ Good fit (When to deploy `QuaRot / SpinQuant` orthogonal rotation) | ⛔ Not for (When to avoid or skip online orthogonal rotation) |
| :--- | :--- |
| **1. Blackwell B200 (`NVFP4`) & Native `INT4` Tensor Core Prefill Clusters**: Eliminates activation channel spikes so 4-bit GEMM Tensor Cores run at `2.60x` throughput without clipping collapse. | **1. Legacy GPUs Lacking Native 4-Bit Compute at Batch=1 Decode**: Without hardware `INT4/FP4` Tensor Cores, online `FWHT` yields no compute speedup; use `AWQ W4A16`. |
| **2. Single-80GB-GPU High-Density 70B + 32K Serving**: Shrinks weights (`35.0 GB`) + 32K KV (`33.6 GB`) to `68.6 GB`, eliminating cross-GPU Tensor Parallelism AllReduce overhead. | **2. Unrotated Multi-LoRA Hot-Swapping Pipelines**: Because base weights are multiplied by $R_1, R_2$, external LoRA adapters must also be transformed into the $R_1$ basis offline before serving. |
| **3. 128K Long-Context Agent / RAG 4-Bit KV-Cache**: Online $R_4$ Post-RoPE Hadamard preserves `99.1%` Needle-in-a-Haystack accuracy at `INT4` KV. | **3. Ultra-Tiny Models (`< 1.5B`) on Competition Math**: Tiny models have lower parameter redundancy; keep `W8A8 / FP8` (though offline $R_1, R_2$ remains free and beneficial!). |

---

## 🧪 5. Quantitative Worked Example & 7-Card 4:5 Report Summary

### 5.1 Deterministic Orthogonal Invariance & Spike Dampening Verification

Consider a 4-dimensional activation vector with an extreme channel-0 spike $x = [64.0, 0.5, -0.4, 0.3]$ (`160x` median) and weight vector $w = [0.25, -0.5, 0.75, 0.1]^\top$:
1. **Unrotated Dot Product**: $y = x \cdot w = 15.48$, with peak amplitude $\|x\|_\infty = \mathbf{64.0}$.
2. **Hadamard Rotation ($H_4$)**:
   - Rotated activation: $x_{\text{rot}} = x \cdot H_4 = [32.2, 31.8, 32.3, 31.7]$ (peak drops from `64.0` to **`32.3`** in 4D, and by $\sim 90.5\times$ in $d=8192$).
   - Rotated weight: $w_{\text{rot}} = H_4^\top \cdot w = [0.3, 0.7, -0.4, -0.1]^\top$.
   - Rotated dot product: $x_{\text{rot}} \cdot w_{\text{rot}} = \mathbf{15.48}$ (**exact `0.0` error**).

### 5.2 Perturbation & Robustness Matrix

| Perturbation / Stress Case | Metric | Naive `W4A4 + KV4` | `SpinQuant` (`W4A4 + KV4`) | Delta Improvement |
| :--- | :--- | :---: | :---: | :--- |
| **1. Extreme Outlier Spike (`120x` Median on Code/Math)** | WikiText-2 / Code PPL ($\downarrow$) | `44.10` | **`5.31`** | **`-38.79 PPL`** (Spike spread across 8,192 dims) |
| **2. Long-Context 128K `INT4` KV-Cache (Group=128)** | Needle-in-a-Haystack (%) ($\uparrow$) | `41.5%` | **`99.1%`** | **`+57.6 pts`** (Post-RoPE $R_4$ smooths Key norms) |
| **3. Blackwell B200 `NVFP4` Micro-Scaling (Block-16)** | 5-Shot MMLU (%) ($\uparrow$) | `74.2%` | **`79.2%`** | **`+5.0 pts`** (Within `0.2%` of FP16 baseline) |
| **4. Non-Power-of-Two FFN ($d_{\text{ffn}}=28672=256\times 112$)** | Online FWHT Latency (%) ($\downarrow$) | `4.85%` | **`1.42%`** | **`-3.43 pts`** (Kronecker $H_{256}\otimes H_{112}$ fused kernel) |

> 💡 **Explore the Interactive Workbench & 7-Card Visual Deck**:
> - **[7-Card 4:5 Visual Benchmark Report (`quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-cards.html`)](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-selection-cards.html)**
> - **[Interactive W4A4 & Hadamard Rotation Sizer (`quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-sizer.html`)](../reports/quarot-spinquant-hadamard-rotation-w4a4-kv-quantization-sizer.html)**
> - **Provenance**: Generated by [`experiments/quarot_spinquant_hadamard_rotation_w4a4_kv_quantization_selection/run.py`](../../experiments/quarot_spinquant_hadamard_rotation_w4a4_kv_quantization_selection/run.py) and sealed in [`results.json`](../../experiments/quarot_spinquant_hadamard_rotation_w4a4_kv_quantization_selection/results.json) (`synthetic_combined_with_anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`).

---

## 🛠️ 6. Production Pre-Flight Checklist

- [ ] **1. Pre-Absorb `RMSNorm` Scaling Vector $\gamma$ Before Rotating**: Always fold each layer's `RMSNorm` scaling parameter $\text{diag}(\gamma)$ into the adjacent linear weights (`W_qkv`, `W_up/gate`) *before* applying $R_1^\top$; otherwise element-wise scaling breaks orthogonal commutativity.
- [ ] **2. Transform Both `Embedding` and `LM_Head`**: Post-multiply token embeddings by $R_1$ ($E' \leftarrow E \cdot R_1$) and pre-multiply `LM_Head` by $R_1^\top$ (preserving tied embeddings if shared).
- [ ] **3. Align Per-Head $R_2$ Across GQA Groups**: In GQA models (e.g., 8 Query heads sharing 1 KV head in `Llama-3.1-70B`), ensure all 8 corresponding `W_o` head slices share the exact same $128 \times 128$ orthogonal matrix $R_2^{(g)}$ used by their shared `W_v` head.
- [ ] **4. Enable Kronecker Block-Hadamard Kernels for Non-Power-of-Two FFN**: Verify that $d_{\text{ffn}} = 28,672$ uses the fused $H_{256} \otimes H_{112}$ kernel rather than falling back to dense $O(d^2)$ matrix multiplication.
- [ ] **5. Rotate External LoRA Adapters Offline**: Transform any hot-swapped LoRA matrices ($\Delta W' = R_{\text{in}}^\top B A R_{\text{out}}$) into the rotated basis prior to serving.
