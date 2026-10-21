# FlashInfer Cake GDP Nemotron 4 Decode/MTP Kernels, NVFP4 4/6 Slot-Mapping Append & Cake Causal Conv1d Backward Routing

> **Upstream Source Commits (2026-10-21)**:
> - `flashinfer-ai/flashinfer` PR `#6883` (`fc1790f4ec33a4df84ae9e6d1dcc9231fde7fd10`): `feat: Add Cake Gated DeltaProduct (GDP) decode & MTP kernels for Nemotron 4 (#6883)`
> - `flashinfer-ai/flashinfer` PR `#6013` (`52fbbd090ca9c85b69c3a94420e360d27779b366`): `feat: support nvfp4_4over6 in slot-mapping paged KV cache append (#6013)`
> - `flashinfer-ai/flashinfer` PR `#6204` (`3821440929ee342fe9f939dbf5186c3e667a851c`): `perf(page): remove .cpu() stream sync from nvfp4 slot-mapping kv append (#6204)`
> - `flashinfer-ai/flashinfer` PR `#6930` (`7c866e85d3bb8095ebb247cdca42a56c2a4cab19`): `perf(causal_conv1d): update Cake backward channel routing for sm103a/sm107a (#6930)`

---

## 1. Architectural Problem Statement

Across **Nemotron 4** Gated DeltaProduct (GDP) hybrid linear-attention serving and **Blackwell / Rubin (`sm_100a` / `sm_103a` / `sm_107a`)** kernel execution, three concrete kernel-level bottlenecks arise:

1. **Multi-Householder State Recurrence & Dual-Regime Occupancy in GDP Decode/MTP**: Gated DeltaProduct updates an FP32 state matrix $S \in \mathbb{R}^{V \times K}$ ($V=64, K=128$) via $n \in \{1, 2, 3, 4\}$ sequential rank-1 Householder reflections per token step $t$. Unfused recurrences repeatedly spill state between registers and memory, while a single CTA partitioning scheme cannot simultaneously saturate SMs at small-batch single-token decode ($T=1, B \times H_V \le 160$) and keep state register-resident across multi-token MTP verification ($T \in [4, 16]$).
2. **NVFP4 `[4.0, 6.0]` Quantization Gap & `.cpu()` Host Sync Bubbles in Paged KV Append**: Standard NVFP4 E2M1 magnitude levels $\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$ map block `amax` to $6.0$ ($sf_6 = \text{fp8\_e4m3}(\text{amax}/(6g))$), leaving the widest quantization step between $4.0$ and $6.0$. Meanwhile, `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` previously called `.cpu()` on CUDA `k_scale`/`v_scale` tensors outside CUDA graph capture, forcing a host-device stream synchronization on every layer during eager prefill.
3. **Suboptimal Causal Conv1d Backward Channel Thresholds on GB300 (`sm_103a`) and Rubin R200 (`sm_107a`)**: `cake_causal_conv1d_train` backward dispatch lacked medium-channel ($512 \le C < 768$ / $1024$) `fold_overlap + fast_roll` and wide-channel ($C \ge 2048$) `roll_prefetch` schedules tuned for GB300 and R200 shared-memory pipelines.

---

## 2. Mathematical Formulation & Kernel Design

```
+--------------------------------------------------------------------------------------------------+
|  FlashInfer Cake GDP Decode/MTP (PR #6883) + NVFP4 4/6 KV Append (#6013/#6204) + Conv1d (#6930)  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [1] Cake GDP Multi-Householder Recurrence (K=128, V=64, FP32 State Pool [pool, HV, 64, 128])    |
|      alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias)),  S <- alpha_t * S                     |
|      for j = 0 .. n-1:                                                                           |
|        k_hat = l2norm(k_{t,j}),  u = S @ k_hat,  S += (sigmoid(b_{t,j}) * (v_{t,j} - u)) (x) k_hat|
|      out_t = scale * (S @ l2norm(q_t))                                                           |
|      * T=1 & B*HV <= 160 -> splitv8_tile8 (8 CTAs/head, 8 V-rows/CTA)                            |
|      * T>=4 or B*HV > 160 -> lanerow (l8r4 / l8r2 / l4r4, 1 CTA/head, register-resident rows)    |
|                                                                                                  |
|  [2] NVFP4 4/6 Block Scale Selection (PR #6013) & Sync-Free Scale Check (PR #6204)               |
|      sf_6 = fp8_e4m3(amax / (6*g)),  sf_4 = fp8_e4m3(sf_6 * 1.5)  (maps amax onto E2M1 level 4.0) |
|      Select sf_4 iff MSE(sf_4) < MSE(sf_6) over 16 elements; compiled in page_nvfp4_4over6       |
|      Remove host .cpu() on CUDA k_scale/v_scale -> rely on in-kernel `trap;` when rows > 0       |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 Cake GDP Multi-Householder State Recurrence & Dual-Body Routing (PR `#6883`)

For each batch slot and value head $h \in [0, H_V)$ ($H = H_V$) over the 16-byte-aligned FP32 state pool `[pool_size, HV, 64, 128]`, token step $t \in [0, T)$ computes:

$$\alpha_t = \exp\!\big(-\exp(A_{\log}) \cdot \operatorname{softplus}(a_t + \text{dt\_bias})\big), \quad \beta_{t,j} = \sigma(b_{t,j})$$

After decaying $S \leftarrow \alpha_t S$, the kernel applies $n \in \{1, 2, 3, 4\}$ sequential Householder updates:

$$\hat{k}_{t,j} = \operatorname{l2norm}(k_{t,j}), \quad u_{t,j} = S \hat{k}_{t,j}, \quad S \leftarrow S + \big(\beta_{t,j}(v_{t,j} - u_{t,j})\big) \otimes \hat{k}_{t,j}$$

and emits $\text{out}_t = \text{scale} \cdot S \operatorname{l2norm}(q_t)$. Setting `disable_state_update=True` computes the exact multi-step output while leaving `initial_state` unmodified in HBM.

`cake_jit.py` dispatches across two specialized kernel bodies:
- **Row-Partition Body (`splitv` / `splitv8_tile8`)**: Selected when $T = 1$ and $B \times H_V \le 160$, launching **8 CTAs per state head** (8 $V$-rows per CTA) to multiply grid occupancy by $8\times$ on Blackwell SMs.
- **Lane-Owns-Rows Body (`lanerow` / `l8r4`, `l8r2`, `l4r4`)**: Selected when $T \ge 4$ or $B \times H_V > 160$, assigning **1 CTA per state head** so the $64 \times 128$ state stays register-resident across all $T \times n$ updates, with static loop-unrolled programs for $n=3, T \in \{1, 4, 8\}$ and runtime programs for $1 \le T \le 16$.

### 2.2 NVFP4 `4/6` MSE Block Scaling & Host-Sync-Free KV Append (PR `#6013` / PR `#6204`)

For each 16-element NVFP4 block, PR `#6013` evaluates both the standard max-6 scale and the `4/6` candidate scale:

$$sf_6 = \operatorname{fp8\_e4m3}\!\left(\frac{\text{amax}}{6 \cdot g}\right), \quad sf_4 = \operatorname{fp8\_e4m3}(sf_6 \cdot 1.5)$$

Choosing $sf_4$ only when $\sum_{i=0}^{15}(\operatorname{mag}_i \cdot sf_4 \cdot g - |x_i|)^2 < \sum_{i=0}^{15}(\operatorname{mag}_i \cdot sf_6 \cdot g - |x_i|)^2$ places exact reconstruction levels at $4.5 \cdot sf_6$ and $2.25 \cdot sf_6$. Isolating `use_4over6=true` into the `page_nvfp4_4over6` JIT module keeps all **51 default kernels in `page` SASS-identical**.

PR `#6204` removes the host-side `float(scale.cpu())` check on CUDA scale tensors in `_validate_positive_scale`, relying on the in-kernel device check (`trap;` when `num_rows > 0` and `scale <= 0` or non-finite) to eliminate one host-device stream sync per attention layer during eager prefill.

### 2.3 Cake Causal Conv1d Backward Channel Routing (PR `#6930`)

`cake_causal_conv1d_train` evaluates descending `channels_at_least` thresholds:
- **`sm_103a` (GB300)**: Routes $512 \le C < 768$ to `fold_overlap=True, fast_roll=True` (`SMEM_TOTAL = 66560 B`) while keeping $C \ge 768$ on the default schedule.
- **`sm_107a` (Rubin R200)**: Routes $512 \le C < 1024$ to `fold_overlap=True, fast_roll=True` (`66560 B`) and $C \ge 2048$ to `fast_roll=True, roll_prefetch=True` (`SMEM_TOTAL = 82944 B`).

---

## 3. Deterministic Verification & Benchmark Summary

Running `python experiments\flashinfer_cake_gdp_nvfp4_4over6_conv1d_bwd\run.py --check` reproduces all invariants and upstream benchmark metrics:

| Dimension | Workload / Target | Baseline | Optimized | Measured Gain |
|---|---|---|---|---|
| **Cake GDP Decode / MTP (`#6883`)** | Nemotron 4 48 shapes (`HV∈{80,20,10,24}, B∈{1..256}, T∈{1,4,8}`) @ **GB200 (`sm_100a`)** | `301.4 us` (`HV=10, B=256, T=8`) | **`75.5 us`** (`l8r2` static $T=8$) | **48/48 wins (`1.01x–3.99x`)** |
| **Cake GDP Decode / MTP (`#6883`)** | Nemotron 4 48 shapes @ **GB300 (`sm_103a`)** | `296.6 us` (`HV=10, B=256, T=8`) | **`75.1 us`** (`l8r2` static $T=8$) | **47/48 wins (`0.98x–3.95x`)** |
| **NVFP4 `4/6` Block Scale (`#6013`)** | 16-element sub-peak clustered block (`±4.5 / ±2.25`) | `MSE = 0.141919` (`sf6 = 1.0`) | **`MSE = 0.000494`** (`sf4 = 1.5`) | **`99.65%` block MSE reduction** |
| **NVFP4 Sync-Free Append (`#6204`)** | SM80 vLLM `Qwen3.6-35B-A3B-NVFP4` (20 layers, 4096-token eager prefill) | `499.4 ms` (`495.5–516.0 ms`) | **`451.7 ms`** (`447.5–464.5 ms`) | **`-47.7 ms` (`-9.55%`, `1.1056x`)** |
| **Cake Conv1d Backward (`#6930`)** | **GB300 (`sm_103a`)** 8 `C=512` shapes / **R200 (`sm_107a`)** 11 routed shapes | `101.3 us` (`R200 B=1,S=250340,C=512`) | **`96.4 us`** (`fold_overlap + fast_roll`) | **GB300 `1.0164x` / R200 `1.0331x` (peak `1.0513x`)** |

---

## 4. Production Guidance & Boundaries

- **When to use**: Nemotron 4 GDP decode and speculative MTP verification on Blackwell (`sm_100a` / `sm_103a`), accuracy-critical NVFP4 paged KV cache append with zero eager host syncs, and GB300 / R200 causal conv1d training.
- **Boundaries**: `cake_gated_delta_product_mtp` requires $K=128, V=64, H=H_V, 1 \le T \le 16, n \in \{1..4\}$ and 16-byte-aligned FP32 state pools; variable-length prefills with $T > 16$ or GQA/GVA head broadcasting should use the chunked varlen GDP prefill path.
