# DeepSeek DeepInfer & the "$38/GPU-Hour / 155T Tokens/Day" Production Ledger: Per-GPU Token Throughput, E/P/D Disaggregation, 100 PiB 3FS KV Cache & Unit Cost Economics

> 🌐 **Language / 语言切换**: **[🇬🇧 English Edition](ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.en.md)** | **[🇨🇳 中文版 (Chinese)](ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.md)** | **[📊 7-Card 4:5 Benchmark Deck (`ai-infra-phase167-benchmark-cards.html`)](ai-infra-phase167-benchmark-cards.html)** | **[⚡ Interactive Visual Workbench (`ai-infra-phase167-visual-workbench.html`)](ai-infra-phase167-visual-workbench.html)** | **[🌐 Live Preview (`turbo998.github.io`)](https://turbo998.github.io/workshops/deepinfer-v41-economics/index.html)**

---

## 📌 Executive Summary

On October 20, 2026, DeepSeek open-sourced its internal production inference framework **[`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer)** (Apache 2.0) alongside the production multi-protocol token parser **`deepseek-recipe`** (MIT), disclosing unprecedented hyper-scale serving telemetry for **`DeepSeek-V4.1-Flash`** (`552B` total MoE parameters, `8B` active in prefill, `16B` active in decode, `1M` context window):
- **Single-Cluster Daily Throughput**: **`155 Trillion Tokens/day`** (`155 × 10^12 tokens/day` ≈ **`1.794 Billion tokens/sec`** cluster-wide, rivaling ByteDance Doubao's 180T/day and `~1.45×` Google's `~107T/day` average).
- **Per-GPU Hourly Revenue**: **`$38 / GPU-hour`** (`~¥258.4 RMB/hr`), measured at **Off-Peak (Valley) API Pricing** across a **`1:1` Prefill-to-Decode GPU pool** (`$76 / GPU-hour` or `~¥516.8 RMB/hr` during weekday peak hours).

Synthesizing primary technical documentation from `deepseek-ai/DeepInfer` ([`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md)), the viral financial & engineering analysis ("【第四种黑猩猩】Why One DeepSeek GPU Earns $38/Hour"), and the independent external reproduction by the `vLLM` team and SemiAnalysis `AgentX` on `GB300 NVL72` (`2026-10-23`), this study establishes a **milestone AI Infrastructure reference on inference efficiency and token unit economics** across three pillars:

1. **Per-GPU Token Production & Kernel-Level Pareto Frontier**:
   - **Prefill (`EP4DP4` @ 4× SM100, 128K context, 32 redundant experts, 140-token SWA bounded replay)**: Cold (uncached) prefill delivers **`60,000` uncached input tokens/s/GPU** at `batch=4,096` tokens/rank, scaling up to **`90,000` uncached tokens/s/GPU** at `batch=65,536` (`~730 ms`/step). Amplified by a **`96%` prefix KV cache hit rate** from the 3FS storage cluster, a single Prefill GPU delivers an **effective billable input throughput of `1,500,000` tokens/s/GPU** (**`25×` physical leverage**, or **`5.4 Billion` input tokens/GPU-hour**, with `400 ms` global average TTFT and `< 1,000 ms` P99).
   - **Decode (`EP64DP64` @ 64× SM100 supernode, 128K context)**: Offline Pareto (`nextn=3`, `75%` draft acceptance $\rightarrow \tau = 3.25$) exceeds **`25,000` output tokens/s/GPU** at `256` requests/rank (`100` tok/s/user) and exceeds **`300` tok/s/user** at `8` requests/rank. In online production (`conc=60` sweet spot), each Decode GPU sustains **`14,000–14,466` output tokens/s/GPU** at **`230` tok/s/user**.
   - **Two Algorithmic Accelerators**: **`DSpark` Confidence-Scheduled Speculative Decoding** (`DEEPINFER_ENABLE_MTP_VARLEN=1`) pairs a lightweight Confidence Head with a fused `TileLang` GPU kernel (cumulative product + global sort top-$k$ + fused remap scatter) to dynamically truncate low-probability draft suffixes—cutting verified draft tokens by **`~30%`** (`5.00` $\rightarrow$ `3.24–3.94`) while preserving accepted length (`3.21` vs `3.32`), yielding **`+5.4% to +11.0%` higher throughput** (`+6%` online average). **`SWA Bounded Replay`** (`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`) exploits the CED/YoCo architecture (layers 0–19 Encoder, layers 20–39 Decoder) to cap suffix recomputation at `128 + 6 = 134/140` tokens instead of `2,560 × 20` decoder tokens—reducing decoder replay by **`94.53%`** ($O(1)$ constant overhead) and nearly **halving total prefill compute**.
2. **E/P/D Physical Disaggregation, 100 PiB 3FS Tiered KV Cache & 4 Closed-Source Load Balancers**:
   - **1:1 P:D Topology & 128-GPU Minimum Unit**: Production separates **Vision Encoder (`VE`, 1 GPU/instance)**, **Prefill (`PE`, commodity 8-GPU nodes at `EP2–8/DP2–8`)**, and **Decode (`CE`, 64–128 GPU supernodes at `EP64–128/DP64–128`)** at a **`1P : 1D` GPU ratio**, with a minimum efficient deployment unit of **`64 Prefill + 64 Decode = 128 GPUs`** (NVIDIA Blackwell `SM100` or Huawei `Ascend 950`), supporting cross-datacenter P/D splits (`~400 Gbps` per 8 GPUs).
   - **`890 Bytes/Token` KV Compression & 100 PiB 3FS Pool**: Cross-Layer KV Sharing + `CSA2` + Main KV `per-16 e4m3` FP4 + Indexer KV `per-32 ue8m0 MXFP4` shrink Global Block KV by **`75%`** to **`890 B/token`** (`87.5%` SSD reduction), stored in a dedicated **`100 PiB` 3FS all-flash cluster** (`10+2` Erasure Coding, query 11 return 10, **`14.4 TB/GPU`**, **`10 TiB/s` peak read / `850 GiB/s` peak write**, **3-day Agent TTL / 7-day Chatbot TTL**, zero-SM **GDR RDMA** swap-in). Short-term SWA states (`20 layers × 134 tokens × 528 B = 1,415,040 B ≈ 1.415 MB/seq`) reside in local host RAM (`180s` TTL, **`>98%` active-session hit rate**). **Decoupling KV storage into 3FS allows any Prefill node to hit any cached prefix globally without routing hotspots.**
3. **Exact "$38/GPU-Hour" Financial Reconstruction & 4 Hidden Moats**:
   - **Prefill GPU Hourly Revenue (`$47.95/hr` Off-Peak)**: Processing `1.5M` total input tok/s = **`5.4 Billion` input tokens/hr**, the **`96%` cached prefix (`5,184M` tokens)** billed at `$0.003/M` (`¥0.02/M`) generates **`$15.55/hr`** (`32.4%` of input revenue), while the **`4%` uncached suffix (`216M` tokens = `60,000` tok/s GPU compute)** billed at `$0.15/M` (`¥1.00/M`) generates **`$32.40/hr`** (`67.6%` of input revenue)—totaling **`$47.95/hr`**.
   - **Decode GPU Hourly Revenue (`$28.05/hr` Off-Peak) & `1P:1D` Blended Average (`$38.00/hr`)**: Delivering `~46.75M` billable output tokens/hr (`~13,000–14,000` tok/s/GPU) at `$0.60/M` (`¥4.00/M`) generates **`$28.05/hr`**. Averaged `1:1` across Prefill and Decode GPUs, cluster revenue equals $\frac{\$47.95 + \$28.05}{2} = \mathbf{\$38.00/\text{GPU-hr}}$ (**`$76.00/GPU-hr`** at peak rates).
   - **Why Renting GPUs Won't Replicate `$38/hr`**: Standard single-node `vLLM` on **H200** yields only `16,600` tok/s/GPU (**`$0.79/GPU-hr`**, losing `-583%` against `$5.40/hr` rental); naive single-node **B300** yields **`$14.88/GPU-hr`** (`39%` of DeepSeek). Only the full stack—**96% cache-hit traffic + 128-GPU supernodes + 100 PiB 3FS + 4 closed-source load balancers (with a mandatory 20% hot-standby quota buffer)**—crosses into the `80%+` gross-margin regime.

---

## 1. Primary Sources & Verification Ledger

| Source ID | Evidence Tier | Primary Repository / Publication | Verified Scope |
| :--- | :--- | :--- | :--- |
| **`SRC-01`** | `implementation` + `author-benchmark` | [`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer) · `README.md` & [`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) (`2026-10-20`) | Official disclosure of `155T tokens/day`, `$38/GPU-hour` off-peak revenue, `96%` cache hit rate, `890 B/token` Block KV, `1.415 MB/seq` SWA KV, `100 PiB` 3FS, DSpark throughput table, SWA 140-token bounded replay, and 4 load balancers |
| **`SRC-02`** | `implementation` | [`deepseek-ai/DeepInfer/examples/sweep/plot_pareto.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/sweep/plot_pareto.py) & `run_sweep.sh` | Pareto throughput equations: $\tau = 1 + \text{nextn} \times a$, $\text{User Speed} = \text{avg\_SPS} \times \tau$, $\text{Output Throughput} = \text{avg\_SPS} \times B_{\text{req}} \times \tau / \text{cp}$ |
| **`SRC-03`** | `author-benchmark` (Independent Repro) | [vLLM Blog: *Serving DeepSeek-V4.1-Flash with PD Disaggregation and Wide-EP on GB300 NVL72*](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep) (`2026-10-23`) & SemiAnalysis `AgentX` | Independent reproduction on 1 `GB300 NVL72` rack (`8P1D`/`7P1D`, Prefill `DEP2` + Decode `DEP32`, `MTP=5`) achieving **`954K` total tok/s/GPU** (`196` tok/s/user P90) and **`$38.96/GPU-hour`** off-peak (`7.9×` single-node baseline), plus H200 (`$0.79/hr`) and naive B300 (`$14.88/hr`) baselines |
| **`SRC-04`** | `financial-breakdown` | "【第四种黑猩猩】Why One DeepSeek GPU Earns $38/Hour" (`https://v.douyin.com/8m31BovOgQs/`) & industry financial audits | Step-by-step reconstruction of `5.4B` input tokens/GPU-hr (`96%` hit `$15.55` + `4%` miss `$32.40` = `$47.95`), `1P:1D` `$38/hr` average, `¥15.35M–¥30.69M RMB/day` cluster revenue, and 4 hidden cost moats |
| **`SRC-05`** | `own-offline` | [`experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py`](../../experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py) | Deterministic zero-dependency Python 3.12 verification suite (`--check`) reproducing Pareto curves, DSpark verification savings, SWA replay bounds, and the `$38/GPU-hr` sensitivity matrix |

---

## 2. Pillar 1: Per-GPU Token Production & Kernel-Level Pareto Frontier

### 2.1 `DeepSeek-V4.1-Flash` Asymmetric CED/YoCo Architecture & Dual-Silicon Support

DeepInfer is purpose-built for **`DeepSeek-V4.1-Flash` (`deepseek-ai/DeepSeek-V4.1-Flash-1017`)**:
- **Asymmetric MoE + CED (Cross-Encoder-Decoder / YoCo) Architecture**: Out of **`552B`** total parameters (`1M` context window), Layers 0–19 form the **Encoder** (interleaving Global Block MLA and Sliding Window Attention `SWA`), while Layers 20–39 form the **Decoder / YOCO Consumer** (using `window=2560` SWA and directly consuming the shared global KV states `ca_state / ica_state` from Layer 19). Consequently, **Prefill activates only `8B` parameters per token, whereas Decode activates `16B` parameters per token**.
- **First-Class NVIDIA Blackwell (`SM100`) & Huawei `Ascend 950` Support**:
  - **NVIDIA SM100 (Blackwell, `CUDA 13`)**: Integrates `FlashMLA`, `DeepGEMM` (`MegaMoE` + `MegaMHC` manifold hyper-connections), `DeepEP`, `TileKernels`, and `TileLang` kernels.
  - **Huawei Ascend 950 (`dav-3510`, `CANN 9.2`)**: Built via `export DEEPINFER_ACCELERATOR=ascend`; Ascend-tailored `DeepGEMM` reaches **`99.8%`** of theoretical dense matrix multiplication peak on Ascend 950DT.
  - **No Legacy GPU Support**: H100/H800 and consumer GPUs are excluded because the stack relies on native sub-byte block-scaled microformats (Main KV `per-16 e4m3` FP4; Indexer KV and MoE weights `per-32 ue8m0 MXFP4`).

### 2.2 Prefill & Decode Per-GPU Throughput Pareto Curves (128K Context on SM100)

Using official 128K (`131,072` tokens) telemetry and the Pareto equations from `examples/sweep/plot_pareto.py`:

$$\tau = 1 + \text{nextn} \times a, \qquad \text{User Speed (tok/s/user)} = \text{avg\_SPS} \times \tau, \qquad \text{Output Throughput (tok/s/GPU)} = \frac{\text{avg\_SPS} \times B_{\text{decode}} \times \tau}{\text{cp}}$$

| Stage & Topology | Batch / Concurrency (`per rank`) | Step Latency / Frequency (`Latency / SPS`) | Physical GPU Compute Throughput (`Uncached / Output tok/s/GPU`) | 96% Cache-Hit Effective Input / User Interactivity | Operational Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prefill (`EP4DP4`)** | `4,096` tokens | `68.3 ms` | **`60,000` uncached tok/s/GPU** | **`1,500,000` total input tok/s/GPU** (`5.4B/hr`) | **Production Interactive Sweet Spot** (Global Avg TTFT `400 ms`, P99 `< 1,000 ms`) |
| **Prefill (`EP4DP4`)** | `16,384` tokens | `221.4 ms` | **`74,000` uncached tok/s/GPU** | **`1,850,000` total input tok/s/GPU** | Balanced Batching Regime |
| **Prefill (`EP4DP4`)** | `32,768` tokens | `399.6 ms` | **`82,000` uncached tok/s/GPU** | **`2,050,000` total input tok/s/GPU** | **Prefill LB Max Engine Quota Cap** (`32,768` tokens) |
| **Prefill (`EP4DP4`)** | `65,536` tokens | `~730.0 ms` | **`90,000` uncached tok/s/GPU** | **`2,250,000` total input tok/s/GPU** | Offline Saturation Peak |
| **Decode (`EP64DP64`)** | `8` reqs/rank | `95.38 SPS` | **`2,480` output tok/s/GPU** | **`310.0` tok/s/user** | Ultra-Low Latency (`>300 tok/s/user`) |
| **Decode (`EP64DP64`)** | `32` reqs/rank | `78.46 SPS` | **`8,160` output tok/s/GPU** | **`255.0` tok/s/user** | High-Interactivity SLA |
| **Decode (`EP64DP64`)** | `60` reqs/rank | `71.79 SPS` | **`14,000–14,466` output tok/s/GPU** | **`230.0–233.3` tok/s/user** | **Production Online Operating Point** (`~14K tok/s/GPU` @ `230 tok/s/user`) |
| **Decode (`EP64DP64`)** | `128` reqs/rank | `49.23 SPS` | **`20,480` output tok/s/GPU** | **`160.0` tok/s/user** | High-Throughput Batching |
| **Decode (`EP64DP64`)** | `256` reqs/rank | `30.77 SPS` | **`25,600` output tok/s/GPU** | **`100.0` tok/s/user** | **Max Offline Throughput** (`>25,000 output tok/s/GPU` @ `100 tok/s/user`) |

> 💡 **Key Insight 1 (The 25× Cache Multiplier)**: While a single SM100 computes `60,000` uncached prefill tokens/sec at interactive latency, bypassing `96%` of prefix tokens via 3FS amplifies billable input throughput by $1 / (1 - 0.96) = \mathbf{25\times}$ to **`1.5 Million` input tokens/s/GPU (`5.4 Billion` tokens/hour)**.

> 💡 **Key Insight 2 (Why 855K Context Barely Slows Decode)**: Profiling on `GB300 NVL72` (`DEP32`, 16 reqs/rank × 6 tokens) shows that at 128K context, **MoE computation + communication accounts for `62%` of step time (`5.54 ms`), while Sparse MLA + Indexer takes only `13%` (`1.16 ms`)**. Expanding context `13.4×` from **64K (`8.86 ms`) to 855K (`9.53 ms`) increases decode step latency by just `+6.7%`**.

### 2.3 `DSpark` Confidence-Scheduled Speculative Decoding: `-30%` Verified Tokens, `+6%` Online Throughput

Instead of blindly verifying all `5` draft tokens (`nextn=5`), `DSpark` (`DEEPINFER_ENABLE_MTP_VARLEN=1`) uses a Confidence Head and a fused `TileLang` GPU kernel (cumulative product + global sort top-$k$ + fused remap scatter) to verify only the prefix that maximizes marginal throughput:

| Concurrency (`per rank`) | Fixed-5 Throughput (`tok/s/rank`) | Fixed-5 Accepted Length | `DSpark` Dynamic Throughput (`tok/s/rank`) | `DSpark` Verified Length | `DSpark` Accepted Length | Verified Tokens Saved | Net Throughput Gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`16`** | `4,891` | `3.31` | `4,816` | `4.44` | `3.24` | `-11.2%` | `-1.5%` |
| **`32`** | `8,964` | `3.31` | `9,111` | `3.65` | `3.17` | **`-27.0%`** | **`+1.6%`** |
| **`48`** | `12,032` | `3.31` | `12,467` | `3.94` | `3.26` | **`-21.2%`** | **`+3.6%`** |
| **`60` (Prod Sweet Spot)** | `13,730` | `3.32` | **`14,466`** | **`3.58`** | **`3.21`** | **`-28.4%`** | **`+5.4%`** |
| **`72`** | `14,962` | `3.29` | **`15,961`** | **`3.24`** | **`3.12`** | **`-35.2%`** | **`+6.7%`** |
| **`96`** | `16,817` | `3.30` | **`18,660`** | **`3.24`** | **`3.15`** | **`-35.2%`** | **`+11.0%`** |

### 2.4 `SWA Bounded Replay` (`KEEP_TOKENS=140`): Cutting Decoder Suffix Recompute by `94.53%`

Because Layers 20–39 (Decoder) consume the global representation from Layer 19 (`ca_state / ica_state`), DeepInfer does not recompute the full `2,560`-token SWA window on a cache hit or SWA miss. Instead, it aligns to `128` tokens (+ `6` speculative backtracking tokens = `134/140` tokens) and restricts attention to the replayed suffix—shrinking decoder recomputation from `20 × 2,560 = 51,200` layer-tokens to `20 × 140 = 2,800` layer-tokens (**`-94.53%`**) and **cutting total prefill compute by ~50%**.

---

## 3. Pillar 2: E/P/D Disaggregation, 100 PiB 3FS KV Cache & 4-Tier Load Balancing

### 3.1 Physical E/P/D Topology & Why `1P : 1D` Holds in Production
- **Vision Encoder (`VE`)**: 1 GPU/instance, routed by machine load and minimum image patches `(W/p)*(H/p)`.
- **Prefill (`PE`, 8-GPU commodity nodes at `EP2–8/DP2–8`)**: Small EP minimizes cross-rank DP straggler synchronization lag (`DEP2` has only **`5.9%` DP lag** at `64.4K tok/s/GPU`, whereas `DEP4` suffers **`13.6%` DP lag** and drops to `53.3K tok/s/GPU`).
- **Decode (`CE`, 64–128 GPU supernodes at `EP64–128/DP64–128`)**: Wide-EP places only **`7` routed experts/GPU** (`1` redundant), keeping MoE weight reads and NVLink `symm_mem` / UB collectives under `< 9 ms`/step.
- **Minimum Efficient Unit**: **`64 Prefill GPUs + 64 Decode GPUs = 128 GPUs`** (`1:1` ratio), with cross-datacenter P/D support (`~400 Gbps` per 8 GPUs).

### 3.2 `890 Bytes/Token` KV Compression & 100 PiB 3FS Storage Pool

| KV Cache Tier | Quantization & Structural Format | Footprint | Storage Tier & Retention TTL | Transfer & Recovery Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Global Block KV** | Cross-Layer KV Sharing + `CSA2` + Main KV `per-16 e4m3` FP4 + Indexer KV `per-32 ue8m0 MXFP4` | **`890 Bytes / token`** (**`-75%` HBM**, **`-87.5%` SSD**) | **`100 PiB` 3FS All-Flash Cluster** (**`14.4 TB/GPU`**, `10+2` EC query-11-return-10; **3-day Agent TTL / 7-day Chatbot TTL**) | **GDR (GPUDirect RDMA)** via NIC PCIe DMA (zero GPU SMs, zero CUDA copy engine); **`10 TiB/s` peak read / `850 GiB/s` write** |
| **SWA State KV** | Stores only 20 Encoder layers × `134` tokens × `528 B` (`MXFP8`), dropping 20 Decoder layers | **`1,415,040 Bytes / seq`** (`~1.415 MB / seq`) | **Local Host RAM (`StateBlock`)** (`~180s` TTL, **`>98%` active-session hit rate**) | Pulled across machines over **`400G VLAN`** via partitioned Entrypoint directory; falls back to `140-token` SWA Bounded Replay |

### 3.3 The Four Closed-Source Load Balancers (`~80%` Online-to-Offline Efficiency)
1. **Prefill LB**: 4 context buckets (`<16K`, `16–64K`, `64–256K`, `>=256K`) + quadratic time-budget solver $\text{cost}(q, c) \le \text{compute\_quota\_limit}$ (capped at `32,768` tokens) synchronized via intra-DP `all-reduce MIN`.
2. **Decode LB**: DualPath 2D balancing of `(active request count, total KV tokens)` pairing short-KV requests with high-KV/low-batch GPUs.
3. **Online EPLB**: Over NVLink `symm_mem`—Prefill uses `4` redundant experts/GPU (`100/1,000` step window, `moe_lb=1.074`); Decode uses `1` redundant expert/GPU (`1,000/10,000` step window, `moe_lb=1.090`).
4. **Autoscaling & `0.8` Quota Guardrail**: Scale-in is blocked whenever quota utilization exceeds **`0.8`** (enforcing a **mandatory `20%+` hot-standby buffer**); Decode scale-in enforces a **1-minute drain grace period** before migrating residual KV caches.

---

## 4. Pillar 3: Exact Reconstruction of the "$38/GPU-Hour & 155T Tokens/Day" Ledger

### 4.1 Official `DeepSeek-V4.1-Flash` Time-of-Use API Pricing
- **Off-Peak (Valley — Weekends all day + Weekday non-peak hours)**:
  - Input Cache Hit: **`¥0.02 / 1M` (`~$0.003 / 1M`)** | Input Cache Miss: **`¥1.00 / 1M` (`~$0.15 / 1M`, `50×` hit price)** | Output: **`¥4.00 / 1M` (`~$0.60 / 1M`)**.
- **Peak (Weekday 09:00–12:00 & 14:00–18:00 CST)**:
  - Exactly **`2.0×` Off-Peak** (`¥0.04 / 1M`, `¥2.00 / 1M`, `¥8.00 / 1M` $\rightarrow$ **`$76.00 / GPU-hour`**).

### 4.2 Step-by-Step Formula Derivation ("Why 1 GPU Earns $38/Hour")
1. **Prefill GPU Hourly Revenue (`$47.95 / Prefill GPU-hr` Off-Peak)**:
   - Physical GPU compute (4% cache miss): $60,000 \text{ tok/s} \times 3,600 \text{ s} = \mathbf{216\text{M uncached tokens/hr}}$.
   - Total billable input at `96%` cache hit: $\frac{216\text{M}}{0.04} = \mathbf{5,400\text{M tokens/hr (5.4B tokens/hr = 1.5M tok/s)}}$, comprising **`5,184M` cached tokens** and **`216M` uncached tokens**.
   - **96% Cache-Hit Revenue**: $5,184\text{M} \times \$0.003/\text{M} = \mathbf{\$15.55/\text{hr}}$ (`32.4%` of input revenue, generated by 3FS storage with near-zero GPU compute!).
   - **4% Cache-Miss Revenue**: $216\text{M} \times \$0.15/\text{M} = \mathbf{\$32.40/\text{hr}}$ (`67.6%` of input revenue).
   - **Total Prefill GPU Revenue**: $\$15.55 + \$32.40 = \mathbf{\$47.95/\text{hr}}$.
2. **Decode GPU Hourly Revenue (`$28.05 / Decode GPU-hr` Off-Peak) & `1P:1D` Cluster Average (`$38.00 / GPU-hr`)**:
   - Effective billable output per Decode GPU: `~46.75M` output tokens/hr (`~13,000–14,000` tok/s/GPU) $\times \$0.60/\text{M} = \mathbf{\$28.05/\text{hr}}$.
   - **Blended `1P : 1D` Per-GPU Hourly Revenue**:
     $$R_{\text{cluster, avg}} = \frac{\$47.95 + \$28.05}{2} = \mathbf{\$38.00 / \text{GPU-hour}} \quad (\text{Off-Peak}) \qquad / \qquad \mathbf{\$76.00 / \text{GPU-hour}} \quad (\text{Peak})$$
3. **155 Trillion Tokens/Day Single-Cluster Scale & Gross Revenue**:
   - At `65.361B` tokens/GPU/day (`(5,400M + 46.75M)/2 × 24`), serving **`155 Trillion` tokens/day** requires **`~2,371` active GPUs** (**`19 × 128-GPU` supernode units = `2,432` GPUs**).
   - **Cluster Daily Revenue**: **`~¥15.35M RMB/day` (`~$2.16M/day`, `~¥5.6B RMB/yr`) at off-peak**, scaling up to **`~¥30.69M RMB/day` (`~¥11.2B RMB/yr`) at peak**.

### 4.3 Cross-Platform Comparison & Why Buying GPUs Won't Make You Rich

| Serving Stack & Hardware | Parallelism & Disaggregation | Total Throughput (`tok/s/GPU`) | User Interactivity (`tok/s/user`) | Off-Peak Revenue (`$/GPU-hr`) | Peak Revenue (`$/GPU-hr`) | Est. GPU Cost (`$/GPU-hr`) | Hardware Gross Margin (`Off-Peak`) | Economic Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Naive Single-Node vLLM (H200)** | 8× H200 SXM (No PD, No Wide-EP) | `16,600` | `100` | **`$0.79 / hr`** | `$1.58 / hr` | `~$5.40 / hr` | **`-583.5%`** | ❌ **Severe Loss** (Cannot cover GPU rent) |
| **2. Naive Single-Node vLLM (B300)** | 8× B300 SM100 (No PD, No Wide-EP) | `317,400` | `100` | **`$14.88 / hr`** | `$29.77 / hr` | `~$8.50 / hr` | `+42.9%` | ⚠️ **Suboptimal** (`39.2%` of DeepSeek benchmark) |
| **3. DeepSeek Prod DeepInfer + 3FS** | 128+ SM100 / Ascend 950 (`1P:1D`, `EP4/EP64`, `DSpark`, `100 PiB 3FS`) | `~756,500` | **`230`** | **`$38.00 / hr`** | **`$76.00 / hr`** | `~$5.00–$8.00 / hr` | **`80.3%–86.8%`** | ✅ **Production Gold Standard** |
| **4. vLLM + Wide-EP + PD (GB300 NVL72)** | 72× GB300 NVL72 (`8P1D`/`7P1D`, `DEP2/DEP32`, `MTP=5`) | **`954,000`** | **`196`** | **`$38.96 / hr`** | **`$77.92 / hr`** | `~$9.00 / hr` | **`76.9%`** | ✅ **Independent External Repro (`7.9×` speedup)** |

---

## 5. Reproducibility Guide

Run the zero-dependency Python 3.12 verification suite to validate all Pareto points, DSpark verification reductions, SWA replay bounds, and the `$38/GPU-hr` ledger:

```bash
python experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py --check
```
