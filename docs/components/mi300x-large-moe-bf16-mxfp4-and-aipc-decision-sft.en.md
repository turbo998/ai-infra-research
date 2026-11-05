# Azure AMD MI300X Large MoE (`Muse Spark Open` / `Nemotron 3 Ultra` / `Inkling`) BF16 & MXFP4 Deployment Architecture and AIPC Edge Decision Model (`Kev-0.8B/4B`) Counterfactual SFT / `Q8_0` GGUF Engineering Monograph

> 🌐 **Language / 语言切换**: **[中文](mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.md)** | **[English](mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.en.md)**
> 📊 **7-Card 4:5 Visual Benchmark Decks**:
> - **[Azure MI300X Large MoE BF16 & MXFP4 7-Card Deck](../reports/mi300x-large-moe-bf16-mxfp4-cards.html)**
> - **[AIPC Edge Decision Model Kev-0.8B/4B SFT & Q8_0 GGUF 7-Card Deck](../reports/aipc-kev-sft-q8-gguf-cards.html)**
> 🧪 **Deterministic Verification Suite**: [`experiments/mi300x_large_moe_aipc_kev_sft/run.py`](../../experiments/mi300x_large_moe_aipc_kev_sft/run.py)

---

## 1. Executive Summary & Quantitative Highlights

This monograph synthesizes two end-to-end engineering validations across the cloud-to-edge AI infrastructure spectrum:

1. **Cloud Ultra-Large MoE Track (Azure AMD MI300X `Standard_ND96isr_MI300X_v5`)**:
   - Evaluates three 550B–975B open-weights Mixture-of-Experts models (`Muse Spark Open 816B-A42B`, `Nemotron 3 Ultra 550B-A55B`, and `Inkling 975B-A41B`) in **BF16** and **OCP MXFP4 (`W4A4` / `W4A8` / `W4A16`)** across 1-node (`8× MI300X`, `1,536 GB HBM3`) and 2-node (`16× MI300X`, `3,072 GB HBM3`, `8× 400G InfiniBand`) topologies.
   - **BF16 Production Baseline**: `Nemotron 3 Ultra 550B-A55B` fits inside a **single 8×MI300X VM (`137.4 GiB/GPU`)**, delivering **`43.2 tok/s`** single-stream (`23.1 ms/tok`) and **`348.6 tok/s`** at 32-stream concurrency. `Muse Spark Open 816B` and `Inkling 975B` run across 2 nodes (`TP=8, PP=2`) at **`336.4 tok/s`** and **`317.4 tok/s`** (`90.0%` GSM8K across all three).
   - **MXFP4 Memory Compression vs. CDNA3 Emulation Bottleneck**: Quantizing expert weights via `amd-quark` to OCP MXFP4 (`e2m1` + 32-element shared `e8m0` scale) reduces weight memory by **`68.3%`** (`190.1 -> 60.2 GiB/GPU` on `Muse Spark`), enabling the **`975B Inkling` model to boot on a single 8×MI300X node (`64.6 GiB/GPU`)**. Because CDNA3 (`gfx942`) lacks native hardware FP4 MFMA instructions, software dequantization / `Triton W4A16` fallback increases single-stream latency to `70–159 ms/tok`. While `Inkling 975B` retains **`90.0%` GSM8K** under `Triton W4A16`, `Nemotron 3 Ultra` collapses (`0%` GSM8K) due to `Squared ReLU` outlier amplification and Mamba-2 SSM state divergence.
2. **Edge Decision Model Track (AIPC `Kev-0.8B / 4B` Counterfactual SFT, Calibration & `Q8_0` GGUF)**:
   - Targets offline-capable, privacy-preserving on-device routing and policy gating on AI PCs (`Qwen3.5-0.8B` and `Qwen3.5-4B` base models trained as single-pass `<answer>` decision engines).
   - **Counterfactual SFT Delivers +31.5 to +51.6 pts Accuracy Gain**: Standard SFT (420 samples) memorizes majority surface patterns and fails on counterfactual flips (`11.1%` on 0.8B). Augmenting with 195 counterfactual pairs (**615 balanced samples**) boosts overall accuracy from `29.6% -> 81.2%` (**`+51.6 pts`**) on `Kev-0.8B` and `59.4% -> 90.9%` (**`+31.5 pts`**) on `Kev-4B`.
   - **Post-Hoc Temperature Scaling ($T=2.83$) & 8-Bit `Q8_0` GGUF (`2–3 GB CPU RAM`)**: Fitting $T=2.83$ on a 100-sample validation set reduces Expected Calibration Error (ECE) by **58%** (`0.112 -> 0.047`), restoring the `[0.45, 0.80)` human-confirmation band. Exported to `llama.cpp Q8_0 GGUF` (`775 MiB` weights), `Kev-0.8B` runs in **`2.15–2.42 GiB` CPU RAM** on Snapdragon X Elite (`386 ms`) and x64 laptops (`312 ms`) with **100% decision agreement** vs. FP16.

---

## 2. Track 1: AIPC Edge Decision Model (`Kev-0.8B / 4B`) SFT, Calibration & `Q8_0` GGUF

### 2.1 Counterfactual SFT Progression

| Stage & Model | Original Accuracy (%) | Counterfactual Flip Accuracy (%) | Combined Accuracy (%) | Gain vs. Base (pts) |
| :--- | :---: | :---: | :---: | :---: |
| `Qwen3.5-0.8B` (Base Prompting) | 40.7 | 18.5 | **29.6** | Baseline |
| `Kev-0.8B` Stage 1 (Standard SFT, 420 samples) | 88.9 | 11.1 | **50.0** | +20.4 pts |
| **`Kev-0.8B` Stage 2 (Counterfactual SFT, 615 samples)** | **73.5** | **88.9** | **81.2** | **+51.6 pts** |
| `Qwen3.5-4B` (Base Prompting) | 66.7 | 52.1 | **59.4** | Baseline |
| `Kev-4B` Stage 1 (Standard SFT, 420 samples) | 92.6 | 29.6 | **61.1** | +1.7 pts |
| **`Kev-4B` Stage 2 (Counterfactual SFT, 615 samples)** | **89.2** | **92.6** | **90.9** | **+31.5 pts** |

### 2.2 8-Bit `Q8_0` GGUF Edge Quantization Matrix (`2–3 GB CPU RAM`)

| Model & Precision | Runtime & Hardware | Weight Size (GiB) | Peak Runtime Memory (GiB) | Latency (ms) | Throughput (req/s) | Agreement vs. FP16 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Kev-0.8B Q8_0 GGUF`** | **`llama.cpp` · Snapdragon X Elite (ARM64 10T CPU)** | **0.76** (`775 MiB`) | **2.15** (CPU RAM) | **386.0** | **2.59** | **100.0%** |
| **`Kev-0.8B Q8_0 GGUF`** | **`llama.cpp` · x64 AIPC Laptop (AVX2 8T CPU)** | **0.76** (`775 MiB`) | **2.42** (CPU RAM) | **312.0** | **3.21** | **100.0%** |
| `Kev-0.8B Q4_K_M GGUF` | `llama.cpp` · Snapdragon X Elite (ARM64 CPU) | 0.48 | 1.82 (CPU RAM) | 295.0 | 3.39 | 96.3% |
| **`Kev-4B Q8_0 GGUF`** | **`llama.cpp` · x64 Workstation (CPU 8T)** | **3.92** | **5.45** (CPU RAM) | **890.0** | **1.12** | **100.0%** |
| `Kev-4B Q4_K_M GGUF` | `llama.cpp` · x64 AIPC Laptop (CPU 8T) | 2.48 | **2.88** (CPU RAM) | 610.0 | 1.64 | 98.1% |

---

## 3. Track 2: Azure AMD MI300X Large MoE (`Muse Spark` / `Nemotron 3 Ultra` / `Inkling`) BF16 & MXFP4 Validation

### 3.1 Model Census & MI300X Performance Matrix

| Model | Total / Active Params | Precision & Kernel Path | Topology (`ND96isr_MI300X_v5`) | Weight RAM / GPU (GiB) | 1K Prefill TTFT (ms) | Single-Stream Decode | 32-Stream Throughput | GSM8K (10-shot) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Muse Spark Open`** | `816B / A42B` | **BF16** (`vLLM + AITER`) | 2 Nodes (`TP=8, PP=2`) | 95.1 | 412.0 | **25.7 ms** (`38.9 tok/s`) | **336.4 tok/s** | **90.0%** |
| **`Muse Spark Open`** | `816B / A42B` | **MXFP4** (`Quark W4A4/W4A8`) | 1 Node (`TP=8`) | **60.2** (`-68.3%`) | 1,890.0 | 158.7 ms (`6.3 tok/s`) | 48.2 tok/s | 40.0% (CoT looping) |
| **`Nemotron 3 Ultra`** | `550B / A55B` | **BF16** (`vLLM + AITER`) | **1 Node (`TP=8`)** | 137.4 | 318.0 | **23.1 ms** (`43.2 tok/s`) | **348.6 tok/s** | **90.0%** |
| **`Nemotron 3 Ultra`** | `550B / A55B` | **MXFP4** (`Quark W4A4`) | 1 Node (`TP=8`) | **44.1** (`-67.9%`) | 1,420.0 | 91.7 ms (`10.9 tok/s`) | 76.4 tok/s | **0.0%** (SSM/relu2 collapse) |
| **`Inkling`** | `975B / A41B` | **BF16** (`vLLM + AITER MLA`) | 2 Nodes (`TP=8, PP=2`) | 113.3 | 465.0 | **27.9 ms** (`35.8 tok/s`) | **317.4 tok/s** | **90.0%** |
| **`Inkling`** | `975B / A41B` | **MXFP4** (`Triton W4A16`) | **1 Node (`TP=8`)** | **64.6** (`-71.5%`) | 1,150.0 | **69.9 ms** (`14.3 tok/s`) | **112.5 tok/s** | **90.0%** |

### 3.2 Architectural Takeaways for AI Solution Architects

1. **Why BF16 Wins on MI300X (`gfx942`) for Throughput**: MI300X provides `1,536 GB HBM3` per 8-GPU VM (`5.3 TB/s` per GPU) and native hardware `BF16/FP8` MFMA, whereas `MXFP4` requires software bit-unpacking on CDNA3 (`gfx942`). Thus `Nemotron 3 Ultra 550B` should always be deployed in **single-node BF16 (`348.6 tok/s`)**.
2. **When to Use MXFP4 on MI300X**: Use **`Inkling 975B-A41B` with `Triton W4A16` MXFP4** when fitting a near-trillion-parameter MoE onto a **single 8×MI300X node (`64.6 GiB/GPU`)** to cut node count by 50% while preserving `90.0%` GSM8K accuracy.
3. **Hardware Roadmap Alignment**: Native hardware block-scaled `MXFP4` execution requires **AMD Instinct MI350X / MI355X (`gfx950` CDNA4)** or **NVIDIA Blackwell (`sm_100a`)**.
