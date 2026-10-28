# SGLang Unified FlashAttention Dispatch, DeepSeek-V4.1 Engram Env Recovery, vLLM PCP/MLA-DCP Speculative Decoding Fixes, KVCR Offload Pruning, and TensorRT-LLM Kimi-K3 Helix-A2A (Phase 118 / Priority 127)

> **Evidence Classification**: Upstream `implementation` (10 public upstream commits, 2026-10-27 to 2026-10-28) + repository `own-offline` deterministic scheduling/memory modeling (`experiments/sglang_fa_dispatch_dsv41_vllm_pcp_mla_dcp_kvcr_trtllm_helix_a2a_phase118/run.py`). No unverified on-GPU hardware re-benchmarking is claimed.

## 1. Architectural Problem Statement & Background

Across production LLM/multimodal serving runtimes (SGLang, vLLM V1, and TensorRT-LLM PyTorch backend), five runtime correctness and efficiency issues were recently addressed:

1. **SGLang Static FlashAttention Import Bypass & Unrecovered Environment Overrides**: `flashattention_backend.py` imported `flash_attn_varlen_func` and `flash_attn_with_kvcache` directly at module scope, allowing call sites to bypass the runtime-selected kernel (such as SM120 FA4); meanwhile, `environ.py`'s `override()` context manager lacked a `try ... finally` block, leaking mutated environment variables if an exception occurred inside the `with` block.
2. **SGLang Realtime/Diffusion Slot Leaks and Redundant Hub API Calls**: `realtime_video_api.py` reserved generation slots before initialization completed, leaking slots on initialization failure; `weights/source.py` queried the remote HuggingFace Hub API even when named weights were already cached locally.
3. **vLLM V1 Speculative Decoding Misalignment Under PCP and MLA DCP**:
   - Under Prefill Context Parallelism (`pcp_size > 1`), `batch_sharded_sampling` indexed draft logits via `req_state_idx // draft_logits_num_shards`, colliding with PCP's cross-rank token partitioning;
   - Under Decode Context Parallelism (`DCP`), ragged speculative-decoding batches (e.g., `query_lens = (3, 1)`) in `flashinfer_mla.py` lacked per-split-row `causal_rows_after` metadata, corrupting causal masking across DCP splits.
4. **vLLM KVCR Adapter Allocation Overhead, ROCm Kimi-K3/Qwen3.8-Flash-Next Enablement, and TensorRT-LLM Kimi-K3 `helix_a2a`**:
   - `kvcr/manager.py` allocated temporary tuples and sets on every `create_store_job` call;
   - TensorRT-LLM needed token-sliced MoE (`helix_a2a`) under Helix CP + Attention DP for Kimi-K3 so each CP rank computes only $1/\text{CP}$ of the tokens' expert projections, while handling 12-head vision towers under non-divisible TP degrees (such as TP16).

---

## 2. Primary Upstream Implementation Analysis (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | Implementation Path & Runtime Mechanism |
|---|---|---|---|
| 1 | `sgl-project/sglang@586355a83b` (`#46695`) | 2026-10-28 | `python/sglang/srt/layers/attention/flashattention_backend.py` & `flash_attention_v4_sm120.py`: Removes static module-level imports of `flash_attn_varlen_func`/`flash_attn_with_kvcache`, routes every FlashAttention call through the selected kernel, adds `mask_mod` to SM120 FA4, and enforces the invariant via an AST unit test. |
| 2 | `sgl-project/sglang@655059819c` (`#46896`) | 2026-10-27 | `python/sglang/srt/environ.py` & `docs/docs/advanced_features/engram.mdx`: Wraps `override()` restoration in `try ... finally` and updates DeepSeek-V4.1-Flash Engram peer-read documentation to use `--attn-dp-size 8`. |
| 3 | `sgl-project/sglang@9694a99d55` (`#46252`) | 2026-10-27 | `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`: Reserves generation slots only after realtime session initialization succeeds and hardens `_cleanup_realtime_session`. |
| 4 | `sgl-project/sglang@5884e4fec6` (`#46679`) | 2026-10-27 | `python/sglang/multimodal_gen/runtime/weights/source.py`: Resolves cached named weights via `snapshot_download(..., local_files_only=True)` before falling back to network downloads on `LocalEntryNotFoundError`. |
| 5 | `vllm-project/vllm@820be160ce` (`#64151`) | 2026-10-28 | `vllm/v1/worker/gpu/spec_decode/speculator.py`: Disables batch-sharded draft logits when `parallel_config.pcp_size > 1`. |
| 6 | `vllm-project/vllm@8697bbae4c` (`#59392`) | 2026-10-27 | `vllm/v1/attention/backends/mla/flashinfer_mla.py`: Computes and passes per-split-row `causal_rows_after` to native FlashInfer MLA DCP calls so ragged speculative decoding batches (`query_lens=(3, 1)`) mask future query tokens accurately. |
| 7 | `vllm-project/vllm@36f5ff3172` (`#63889`) | 2026-10-27 | `vllm/v1/kv_offload/tiering/kvcr/manager.py`: Eliminates generator-tuple and `set()` allocations in `_resolve_pin` while verifying `type(descriptor.element_index) is int`. |
| 8 | `vllm-project/vllm@bb7ea479c6` (`#63409`) | 2026-10-27 | `vllm/models/kimi_k3/amd/mla.py`: Passes explicit zero positions to the ROCm fused MLA decode `q-prep` kernel for Kimi-K3. |
| 9 | `vllm-project/vllm@5314c58281` (`#63332`) | 2026-10-27 | `recipes/models/Qwen/Qwen3.8-Flash-Next.yaml`: Adds an AMD Instinct MI355X Quark MXFP4 recipe (TP1/TP2) for Qwen3.8-Flash-Next (125B main model + 51B N-gram embeddings, 6B active parameters per token). |
| 10 | `NVIDIA/TensorRT-LLM@8786940e8b` (`#20597`) | 2026-10-28 | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` & `modeling_kimi_k3_vl.py`: Implements `helix_a2a` token-sliced MoE under Helix CP + Attention DP for Kimi-K3 (slicing tokens across CP ranks via `_helix_cp_pad` and All-to-All), and replicates the 12-head vision tower (`tp=1`) when attention TP (e.g., TP16) does not divide head count. |

---

## 3. Key Formulas, Data Layout & Offline Validation Results

### 3.1 TensorRT-LLM `helix_a2a` Token-Sliced MoE & MLA DCP Ragged Causal Offset
1. **Token-Sliced MoE Compute Under Helix CP**: With $T$ total tokens and Helix CP degree $C_{\text{helix}}$, `helix_a2a` pads and slices tokens across CP ranks via All-to-All so each rank processes:
   $$T_{\text{rank}} = \left\lceil \frac{T}{C_{\text{helix}}} \right\rceil$$
   At $T = 4096, C_{\text{helix}} = 8$, $T_{\text{rank}} = 512$, reducing per-rank MoE compute by **$8\times$** (`helix_a2a_moe_compute_reduction_factor = 8`).
2. **MLA DCP Ragged `causal_rows_after`**: For request $i$ with query length $q_i$, query row $j \in [0, q_i - 1]$ has:
   $$\text{causal\_rows\_after}(i, j) = q_i - 1 - j$$
   subsequent query tokens in the same request, yielding `[2, 1, 0, 0]` for `query_lens = (3, 1)`.

### 3.2 Offline Verification Summary (`run.py --check`)

- `fa_bypassed_calls`: `0` (100% of FlashAttention calls routed through selected kernel)
- `env_override_leaks_on_exception`: `0` (`try/finally` restores environment state on exception)
- `leaked_realtime_slots_on_init_failure`: `0` (slots reserved strictly post-initialization)
- `batch_sharded_draft_logits_under_pcp`: `false` (disabled whenever `pcp_size > 1`)
- `mla_dcp_ragged_causal_rows_after`: `[2, 1, 0, 0]` (exact ragged causal tail counts for `(3, 1)`)
- `kvcr_containers_saved_per_job`: `3` (eliminates 3 container allocations per KVCR store job)
- `qwen38_flash_next_mxfp4_main_gib`: `65.48` (`3.5557x` compression vs. BF16 for the 125B main model)
- `helix_a2a_moe_compute_reduction_factor`: `8` (`512` tokens/rank at `CP=8`)
- `kimi_k3_vision_effective_tp_at_tp16`: `1` (replicated vision tower avoids `12 % 16 != 0` failure)

---

## 4. Exercises & Falsifiable Questions for Cloud Solution Architects

1. **Capacity Sizing Exercise**: Qwen3.8-Flash-Next combines a 125B-parameter main model with 51B N-gram embeddings (activating 6B parameters per token). With Quark MXFP4 (`0.5625 B/param` including E8M0 block scales) on the 125B main model and FP8 (`1.0 B/param`) on the 51B N-gram embeddings, compute the total weight + embedding footprint on a single AMD Instinct MI355X (`288 GiB` HBM3e, TP1) and the remaining HBM budget for KV cache. (*Answer*: $65.48\text{ GiB} + 47.50\text{ GiB} = 112.98\text{ GiB}$, leaving $\approx 175.02\text{ GiB}$ for KV cache and activation buffers).
2. **Falsifiable Hypothesis**: Does replicating Kimi-K3's 12-head vision tower (`tp=1`) under TP16 bottleneck throughput? No: the vision tower accounts for `< 2%` of total parameters, and under Attention-DP each rank independently encodes distinct images without requiring 16-way tensor-parallel all-reduce across every vision layer.
