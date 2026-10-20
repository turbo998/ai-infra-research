# veRL Low-Precision QAT Rollout In-Place Weight Refit (`vllm_fp4_utils.py` Zero-Copy Storage Aliasing) and Tail-Aware Coarse-Grained Top-K On-Policy Distillation (`forward_kl_topk_tail`)

> **中文版**: [verl-qat-nvfp4-mxfp8-rollout-tail-kl-distillation.md](verl-qat-nvfp4-mxfp8-rollout-tail-kl-distillation.md) | **Experiment**: [`experiments/verl_fp4_refit_tail_kl_sim/README.md`](../../experiments/verl_fp4_refit_tail_kl_sim/README.md)

### Executive Summary

veRL introduces two tightly coupled post-training algorithms and systems mechanisms: (1) For low-precision QAT rollout refit (commit `8f016846e3`), **`_mxfp4_staging_data` / `_nvfp4_staging_data` zero-copy byte-view aliasing (`data.flatten().view(dtype).reshape(shape)`) paired with `_replace_parameter_in_place`** reloads 61 MoE layers (32 routed experts/GPU, `40,992 MiB` packed 4-bit weights) without allocating a duplicate copy of the expert weights, staging only the block scales (**`1/16 = 6.25%` or `2,562 MiB` for MXFP4; `1/8 = 12.5%` or `5,124 MiB` for NVFP4**) while keeping every CUDA-Graph-captured tensor address intact; (2) For On-Policy Distillation (commit `5a8dd519ab`), **`forward_kl_topk_tail` (`compute_tail_bucket_kl` and `compute_tail_aware_logit_gradient`)** eliminates the negative-loss clamping pathology of truncated `forward_kl_topk` when student top-$k$ mass exceeds teacher top-$k$ mass ($Q_k > P_k$), adding a single aggregate tail bucket ("ghost token") with a one-sided student epsilon guard (`tail_mass_eps = 1e-6`) to recover an exact non-negative coarse-grained KL lower bound (turning `-0.19125` clamped to `0.0` into **`+0.205276`** at $P_k=0.70, Q_k=0.92$) and an analytic vocabulary-sharded logit gradient with zero extra teacher RPC payload.

### Upstream Source Responsibility & Locations

| Module & File | Upstream Repo & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `verl/utils/qat/compressed_tensors.py` | `volcengine/verl` commit `8f016846e3010e225072a947270dc0c1b98c1cd0` / blob `3408132742c3291a760e4a484d9090dae61545d4` | Defines `SUPPORTED_SCHEMES` (`FP8_BLOCK`, `MXFP8`, `MXFP4`, `NVFP4`), `DEFAULT_IGNORE` regexes, and `CompressedTensorsWeightQuantizer` for trainer-to-rollout weight export. |
| `verl/utils/vllm/vllm_fp4_utils.py` | `volcengine/verl` commit `8f016846e3010e225072a947270dc0c1b98c1cd0` / blob `184e96191aef2955b201eda3ff88905abded8fcf` | Implements `install_mxfp4_mxfp8_compressed_tensors_patch` (`flashinfer_cutlass` default), zero-copy weight aliasing `_mxfp4_staging_data`, and CUDA-Graph-safe `_replace_parameter_in_place`. |
| `verl/trainer/distillation/tail_kl.py` | `volcengine/verl` commit `5a8dd519ab32bfd36d2509a137f830ba79e89c54` / blob `12be58de3657be1748c0f62750691ea31a5f3343` | Implements one-sided epsilon-guarded `compute_tail_bucket_kl` and vocab-sharded analytic logit gradient `compute_tail_aware_logit_gradient`. |
| `docs/algo/opd.md` | `volcengine/verl` commit `5a8dd519ab32bfd36d2509a137f830ba79e89c54` / blob `a844cc30588eb5182fda8c080914c576b401c40a` | Formalizes GKD OPD, PG OPD (`k1` reverse-KL estimator), and `forward_kl_topk_tail` coarse-grained KL lower-bound properties. |

### Core Implementation Path, Memory Layout & Synchronization Contracts

1. **Zero-Copy Storage Aliasing for CUDA-Graph-Safe MoE FP4 Refit (`vllm_fp4_utils.py:16-30, 238-319`)**:
   - Under `FLASHINFER_CUTLASS_MXFP4_MXFP8`, `DEEPGEMM_MXFP4`, and `MARLIN`, the post-processed kernel weight tensors occupy the exact same byte count as the `uint8` checkpoint layout (`data.numel() * data.element_size() == shape.numel() * dtype.itemsize`).
   - `_mxfp4_staging_data` aliases the live storage via `data.flatten().view(dtype).reshape(shape)` so `load_weights` writes directly into the live buffer at zero memory cost. Only the block scales (`_MXFP4_SF_BLOCK = 32` at `1/16` weight bytes, or `_NVFP4_SF_BLOCK = 16` at `1/8` weight bytes) allocate temporary staging tensors before `_replace_parameter_in_place` copies transformed results back into `param.data`, preserving all CUDA-Graph-captured addresses.
2. **Routing `MXFP4xMXFP8` to FlashInfer CUTLASS (`vllm_fp4_utils.py:48-144`)**:
   - `install_mxfp4_mxfp8_compressed_tensors_patch` overrides vLLM 0.29's `CompressedTensorsMoEMethod.get_moe_method` when `weights=mxfp4` and `input_activations=mxfp8`, selecting `flashinfer_cutlass` (`kMxfp8Dynamic`) by default to bypass a numerical bug in TRTLLM's `MXFP4xMXFP8` path on Qwen3.5.
3. **Coarse-Grained Tail-Bucket Forward KL (`tail_kl.py:18-70`, `opd.md:430-502`)**:
   - Let $P_k = \sum_{v \in T_k} \nu(v\mid s_t)$ and $Q_k = \sum_{v \in T_k} \pi_\theta(v\mid s_t)$. Adding the single non-top-$k$ tail bucket yields:
     $$\mathcal{L}_{\mathrm{tail}}^{(k)} = \sum_{v \in T_k} p_v \log\frac{p_v}{q_v} + (1 - P_k)\log\frac{\max(1 - P_k, \text{tiny})}{\max(1 - Q_k, \epsilon_{\text{tail}})}$$
   - Its analytic gradient with respect to student logit $z_j$ is $(P_k + \mathbb{I}_{1-Q_k > \epsilon_{\text{tail}}} P_{\text{tail}}) q_j - p_j$ for $j \in T_k$, and $\left(P_k - \mathbb{I}_{1-Q_k > \epsilon_{\text{tail}}} P_{\text{tail}} \frac{Q_k}{Q_{\text{tail}}}\right) q_j$ for $j \notin T_k$, satisfying $\sum_{j \in V} \partial \mathcal{L}_{\mathrm{tail}}^{(k)} / \partial z_j = 0$.

### Deterministic Verification (`experiments/verl_fp4_refit_tail_kl_sim`)

Executed via `python experiments/verl_fp4_refit_tail_kl_sim/run.py --check` (`experiments/verl_fp4_refit_tail_kl_sim/results/summary.json`):

| Evaluation Dimension / Scheme | Metric 1 | Metric 2 | Metric 3 | Metric 4 |
| :--- | :---: | :---: | :---: | :---: |
| **Case A: Overconfident Student Top-$k$** ($P_k=0.70, Q_k=0.92$) | Raw Top-$k$ KL: `-0.19125` (Clamped to `0.0` in legacy mode) | Tail Bucket KL: `+0.396527` | **Coarse-Grained Total KL: `+0.205276` (Non-negative)** | Tail Gradient L1 Norm: `0.2200` (FD error `0.0`) |
| **Case B: Underconfident Student Top-$k$** ($P_k=0.88, Q_k=0.55$) | Raw Top-$k$ KL: `+0.413848` | Tail Bucket KL: `-0.158611` | **Coarse-Grained Total KL: `+0.255237`** | Tail Gradient L1 Norm: `0.3300` ($\sum \nabla_z = 0$) |
| **Refit: `verl_MXFP4xMXFP8_zero_copy_alias`** | Live 4-bit MoE Weights: `40,992.0 MiB/GPU` | **Peak Refit Staging: `2,562.0 MiB/GPU`** | **Staging/Weight Ratio: `0.0625` (`1/16`)** | CUDA Graph Preserved: `true` (`-94.4%` vs naive clone) |
| **Refit: `verl_NVFP4_W4A4_zero_copy_alias`** | Live 4-bit MoE Weights: `40,992.0 MiB/GPU` | **Peak Refit Staging: `5,124.0 MiB/GPU`** | **Staging/Weight Ratio: `0.1250` (`1/8`)** | CUDA Graph Preserved: `true` (`-88.9%` vs naive clone) |

### Architect Design Exercises & Falsifiable Questions

1. **Why must `forward_kl_topk` and `forward_kl_topk_tail` never be combined with `use_policy_gradient=true` (`opd.md:490-502`)?**
   - **Answer**: Policy-gradient updates act solely through $\nabla_\theta \log \pi_\theta(y_t \mid s_t)$ of the single sampled token $y_t$, whereas `forward_kl_topk_tail` sums divergence across all $k$ teacher tokens plus the tail bucket. If the student already matches the teacher on $y_t$ but diverges on non-sampled top-$k$ tokens, a positive forward KL used as a PG reward will erroneously push $\pi_\theta(y_t \mid s_t)$ away from $\nu(y_t \mid s_t)$.
2. **Falsifiable Hypothesis**: Does vLLM's `Triton` MXFP4 MoE backend support in-place weight refit via `_replace_parameter_in_place`?
   - **Verification**: `vllm_fp4_utils.py:196-198` explicitly excludes Triton because it wraps weights in non-Parameter objects and stores scales on the quant method rather than publishing through `replace_parameter`.
