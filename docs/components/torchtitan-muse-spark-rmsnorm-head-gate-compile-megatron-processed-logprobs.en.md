# TorchTitan Muse Spark Regional Compile Budgets (`muse_spark_rmsnorm` / Head-Gate) and Megatron-LM Raw Prompt Log-Probs under `processed_logprobs`

> **中文版**: [torchtitan-muse-spark-rmsnorm-head-gate-compile-megatron-processed-logprobs.md](torchtitan-muse-spark-rmsnorm-head-gate-compile-megatron-processed-logprobs.md)

## 1. First-Hand Source Evidence

| System / Repository | Commit / Date | Core Source Paths | Evidence Class |
|---|---|---|---|
| **TorchTitan** (`pytorch/torchtitan`) | `e923bfc7fa6f78da7f1bbce48133457a9eb77b92` (`2026-10-20`, PR `#5434`) | `torchtitan/models/muse_spark/model.py`, `tests/unit_tests/gpu/models/muse_spark/test_attention_compile.py` | `implementation` + `author-benchmark` |
| **TorchTitan** (`pytorch/torchtitan`) | `08b4c0ebdfc9d2f733020abdb829c76c9d8e3cb9` (`2026-10-20`, PR `#5458`) | `torchtitan/models/muse_spark/model.py`, `tests/unit_tests/gpu/models/muse_spark/test_attention_compile.py` | `implementation` + `author-benchmark` |
| **Megatron-LM** (`NVIDIA/Megatron-LM`) | `4f86e2c1ab17b29a2fd54d64819a0c5fe437a5a3` (`2026-10-19`, PR `#8162`) | `megatron/core/inference/contexts/dynamic_context.py`, `tests/unit_tests/inference/contexts/test_dynamic_context.py` | `implementation` |

---

## 2. TorchTitan Muse Spark: Per-Head Attention Norms, Output Gate, and Residual-Stream RMSNorm Compilation (`#5434`, `#5458`)

### 2.1 Bandwidth Starvation of Eager ATen `rms_norm` on 64-Wide Rows (`#5434`)
In `MuseSparkAttention` (Muse Spark Open 816B-A42B), each layer normalizes per-head queries (`[T, 128, 64]`) and keys (`[T, 16, 64]`), scales queries by `scale_query_by`, and runs `contiguous -> output_norm -> reshape([T, 8192]) -> * sigmoid(output_gate_TO)` on the attention output before `self.wo`.

On 1x GB300 at $T = 4096$:
- Eager ATen `rms_norm` on 64-wide rows takes **$403\ \mu\text{s}$** forward for `q + k` vs. an ideal bandwidth floor of **$\sim 21\ \mu\text{s}$** (**$5.21\%$ efficiency**);
- Including separate scale, reshape, and gate kernels, forward+backward glue costs **$794\ \mu\text{s}$** for q/k (ideal $42\ \mu\text{s}$) and **$814\ \mu\text{s}$** for output (ideal $66\ \mu\text{s}$), totaling **$1,608\ \mu\text{s}/\text{layer}$** (ideal $108\ \mu\text{s}$).

Commit `e923bfc7` introduces `@local_compile("muse_spark_attention_qk_norm")` and `@local_compile("muse_spark_attention_output")`, keeping `self.wo` (with TP reduction and its own remat region) outside. FMA contraction lowers BF16 max relative error vs. FP64 (`4.3e-3 -> 2.3e-3` on output gate backward) while cutting 2-layer latency by **$-6.7\%$ to $-7.2\%$ in training**, **$-10.8\%$ to $-11.5\%$ in prefill**, and **$-7.3\%$ at decode-256**.

### 2.2 Why `muse_spark_rmsnorm` Requires `recompile_limit=16` (`#5458`)
Commit `08b4c0eb` compiles `MuseSparkRMSNorm.forward` under `@local_compile("muse_spark_rmsnorm", batch_invariant=False, recompile_limit=16)`:
- Because `attention_norm`, `ffn_norm`, `post_attention_norm`, `pre_expert_norm`, and the final norm share `MuseSparkRMSNorm.forward`, the region serves **4 call shapes** (FP32 stream with weight, BF16 without weight, 4096-wide expert norm, final norm) $\times$ **grad mode** $\times$ **dynamic $T$**, producing **12 compiled graphs** across training and inference. Raising `recompile_limit` from the default `8` to `16` prevents dynamo recompile exhaustion.
- Across the full PR 1–4 stack on GB300, 2-layer latency drops from **$42.13 \to 33.17\text{ ms}$ ($-21.27\%$)** for 4k training, **$59.31 \to 45.17\text{ ms}$ ($-23.84\%$)** for 16k prefill, and **$1.55 \to 1.20\text{ ms}$ ($-22.58\%$)** for decode-256.

---

## 3. Megatron-LM: Preserving Raw Prompt Log-Probs under `processed_logprobs` (`#8162`)

### 3.1 Why Prompt Rows Must Not Use `sampling.log_probs_kernel`
Previously, when `logprobs_mode='processed_logprobs'`, `DynamicInferenceContext.calculate_log_probs_tensors` mapped every row in a mixed prefill/decode step to its parent request via `repeat_interleave(active_query_lengths)` and ran `sampling.log_probs_kernel` (applying temperature, top-k, and top-p) across all prompt tokens. Because prompt tokens are user-supplied rather than sampled from the filtered distribution, top-k/top-p masking corrupted prompt log-probs and prompt `top_n_logprobs` with $-\infty$.

### 3.2 Sampled-Row Indexing and Partial Chunked-Prefill Exclusion
Commit `4f86e2c1` computes `log_probs = F.log_softmax(logits_squeezed, dim=-1)` for all rows first, locates each request's final row via `new_token_idx = active_query_lengths_gpu.cumsum(0) - 1`, and checks `self.get_index_of_chunked_prefill_request(safe=True)`:
- If the last active request is an incomplete chunked prefill (`chunked_idx == self.total_request_count - 1`), its final row scores its **next prompt token** rather than an engine-sampled token, so `num_sampled` is decremented to `n_active - 1`.
- Only `sampled_idx = new_token_idx[:num_sampled]` are overwritten with `_processed_log_probs`, and out-of-support sampled tokens fall back to the tempered log-prob `gathered - scaled.logsumexp(dim=1)`.

---

## 4. Reproducible Offline Verification

```bash
python experiments/torchtitan_muse_spark_compile_megatron_logprobs/run.py --check
```
