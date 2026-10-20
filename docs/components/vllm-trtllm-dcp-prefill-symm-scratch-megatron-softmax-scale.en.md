# vLLM FlashInfer DCP `trtllm-gen` Prefill, NCCL Symmetric-Memory Scratch Cloning Isolation, and Megatron-LM `softmax_scale` 3-Tier Precedence

**Language / 语言**: [English](vllm-trtllm-dcp-prefill-symm-scratch-megatron-softmax-scale.en.md) | [中文](vllm-trtllm-dcp-prefill-symm-scratch-megatron-softmax-scale.md)

> **Evidence Tier**: `implementation` (first-hand upstream commit inspection) + `own-offline` (Python 3.12 standard-library deterministic verification via `experiments/vllm_trtllm_dcp_symm_scratch_megatron_scale/run.py --check`). No live multi-GPU hardware profiling is claimed.

---

## 1. Executive Summary & Architectural Findings

Across long-context Decode Context Parallelism (DCP), NVLink symmetric-memory collectives, and Multi-Latent Attention (MLA) training/inference alignment, three upstream commits on 2026-10-20 resolved critical data-path and numerical contract bugs:

1. **vLLM FlashInfer DCP Prefill Context Pass via `trtllm-gen` & Empty-Shard Masking ([`vllm-project/vllm@7b569ec895`](https://github.com/vllm-project/vllm/commit/7b569ec8953cf8099638a6d2ee1e915b75867c3a))**:
   - Adds `TrtllmDCPContextWrapper` in `vllm/v1/attention/backends/flashinfer.py` to execute the DCP prefill context pass on `trtllm_batch_context_with_kv_cache` when `use_dcp` is enabled, KV cache is unquantized (`KVQuantMode.NONE`), `window_left == -1`, and `logits_soft_cap` is disabled.
   - **Workspace Carve-Out Gate**: `TrtllmDCPContextWrapper.fits_workspace(qo_indptr_cpu, num_qo_heads * dcp_world_size)` mirrors the FlashInfer `trtllm-gen` launcher LSE softmax statistics buffer requirement:
     $$\text{stats\_bytes} = (H_q \cdot W_{\text{dcp}}) \times B \times \operatorname{round\_up}(\max L_q, 256) \times 8 + 1\text{ MiB}$$
     falling back cleanly to `BatchPrefillWithPagedKVCacheWrapper` whenever `stats_bytes + (1 << 20)` exceeds `_get_trtllm_workspace_buffer().nbytes`.
   - **Empty Local Context Shard (`seq_lens == 0`) NaN Prevention**: `trtllm-gen` writes neither `out` nor `lse` for requests with zero local context tokens (`_has_empty_ctx == True`). Using `torch.empty_like(q)` left uninitialized `NaN`/`Inf` values in `out` that caused $0 \times \text{NaN} = \text{NaN}$ during `cp_lse_ag_out_rs` / `dcp_a2a_lse_reduce`. The fix initializes `out = torch.zeros_like(q)` when `self._has_empty_ctx` is true and invokes `mask_dcp_empty_shards_(lse, self._seq_lens, self._cum_seq_lens_q)` to set empty-shard `lse = -inf` (`0.0` combine error vs `NaN` on legacy).

2. **vLLM NCCL Symmetric-Memory Collective Scratch Cloning ([`vllm-project/vllm@8122776fba`](https://github.com/vllm-project/vllm/commit/8122776fba2dcfae2973a0659db2da6ac688306b))**:
   - In `vllm/distributed/device_communicators/cuda_communicator.py`, `_get_symm_scratch` caches reusable symmetric-memory tensors keyed by `(role, dtype, device)` (`"rs_out"` for `_reduce_scatter_symm_mem` when `output is None`, and `"ag_out"` for `_all_gather_symm_mem`).
   - Returning the internal scratch buffer directly exposed callers to silent in-place overwrite whenever a subsequent collective with the same role key executed while the earlier result remained live (`30.75` L1 overwrite error in two-step deterministic replay). Returning `output.clone()` (when `scratch_output` is true) and `symm_output.clone()` isolates caller tensor lifetimes (`0.0` error) while keeping caller-supplied `output` tensors zero-copy.

3. **Megatron-LM `Attention.get_softmax_scale` 3-Tier Precedence & YaRN MLA Composition ([`NVIDIA/Megatron-LM@0156c21373`](https://github.com/NVIDIA/Megatron-LM/commit/0156c213738eaf718f7fba2a77a3b79fa4f4ef97))**:
   - Introduces `Attention.get_softmax_scale(query_tensor, head_dim_index=-1)` in `megatron/core/transformer/attention.py`, enforcing the exact resolution order `self.softmax_scale` (MLA) $\to$ `self.config.softmax_scale` $\to$ $d_q^{-0.5}$ across `flash_decode` and `flash_decode_and_prefill`.
   - Fixes `MLASelfAttention.__init__` in `megatron/core/transformer/multi_latent_attention.py` to multiply YaRN `mscale * mscale` by `self.config.softmax_scale` when provided instead of hardcoding `1 / math.sqrt(self.q_head_dim)`, eliminating a `1.316359x` logit scaling discrepancy at $d_q = 192$ and `config.softmax_scale = 0.095`.

---

## 2. Implementation Data Path & Workspace Sizing Matrix

| Scenario (`H_q * W_dcp = 64`, Workspace = `8 MiB`) | `batch_size` | `max_q_len` | `round_up(max_q_len, 256)` | `stats_bytes` (`B`) | `required_bytes` (`B`) | Selected Wrapper |
|---|---:|---:|---:|---:|---:|---|
| `B=4, max_q=1024` | 4 | 1024 | 1024 | `2,097,152` | `3,145,728` (`3.00 MiB`) | `TrtllmDCPContextWrapper` |
| `B=8, max_q=2048` | 8 | 2048 | 2048 | `8,388,608` | `9,437,184` (`9.00 MiB`) | `BatchPrefillWithPagedKVCacheWrapper` |
| `B=8, max_q=3600` | 8 | 3600 | 3840 | `15,728,640` | `16,777,216` (`16.00 MiB`) | `BatchPrefillWithPagedKVCacheWrapper` |
| `B=16, max_q=2048` | 16 | 2048 | 2048 | `16,777,216` | `17,825,792` (`17.00 MiB`) | `BatchPrefillWithPagedKVCacheWrapper` |

---

## 3. Deterministic Offline Verification (`own-offline`)

Verified via `python experiments/vllm_trtllm_dcp_symm_scratch_megatron_scale/run.py --check` (`experiments/vllm_trtllm_dcp_symm_scratch_megatron_scale/results/summary.json`):

| Evaluation Dimension | Legacy Counterfactual | Verified Implementation | Quantitative Result / Invariant |
|---|---|---|---|
| **DCP Empty Context Shard (`seq_len=0`)** | `empty_like(q)` + unmasked `lse` $\to$ `NaN` | `zeros_like(q)` + `mask_dcp_empty_shards_` (`lse=-inf`) | Empty shard weight `0.0`, combined output `1.10` (`0.0` error) |
| **Symmetric Memory Scratch (`rs_out` / `ag_out`)** | Returns internal scratch alias (`30.75` L1 overwrite error) | Returns `.clone()` on internal scratch | `0.0` L1 overwrite error on retained activations |
| **Megatron-LM MLA `softmax_scale` (`d_q=192`)** | Ignores `config.softmax_scale=0.095` (`0.135234`) | `mscale^2 * config.softmax_scale` (`0.17801615`) | Fixes `1.316359x` logit scale mismatch |

---

## 4. Architect Design Exercise & Falsifiable Follow-Up

1. **Capacity Exercise**: For `dcp_world_size = 8`, `num_qo_heads = 16` per rank (`128` total), `B = 8`, and `max_q_len = 4096`, what minimum `_get_trtllm_workspace_buffer()` size prevents fallback to `BatchPrefillWithPagedKVCacheWrapper`?
   - **Answer**: $128 \times 8 \times 4096 \times 8 + 1\text{ MiB} = 33\text{ MiB}$.
2. **Falsifiable Follow-Up**: If cross-layer communication overlap is enabled outside Dual-Batch Overlap (DBO), can MoE combine reuse of `_get_symm_scratch` still race across streams without per-stream cache keys?
