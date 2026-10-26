# SGLang Scheduler WeakRef, HiCache Empty Aux Skip, Multimodal Precomputed Hash/Offset, Single-Pass VAE Seam Blending, vLLM Rust P/D Cache Hits, NIXL DCP Interleave, DeepSeek-V4 MXFP8/MXFP4 & TensorRT-LLM PrimTS Sparse MLA Deep Dive (Phase 90)

> **中文版本**: [sglang-weakref-hicache-vae-vllm-pd-dcp-mxfp8-trtllm-primts-phase90.md](sglang-weakref-hicache-vae-vllm-pd-dcp-mxfp8-trtllm-primts-phase90.md)  
> **Evidence Tier**: `implementation` (first-hand commit inspection) + `own-offline` (deterministic Python 3.12 stdlib model in `experiments/sglang_weakref_hicache_vae_vllm_pd_dcp_mxfp8_trtllm_primts_phase90/run.py`). No live GPU/RDMA hardware benchmarks are claimed.

## 1. First-Hand Code Evidence & Implementation Paths

1. **SGLang Scheduler Weak `__req__` Self-Reference** (`sgl-project/sglang` PR `#45923`, commit `3a62cdfb3d`):
   - Replaces the strong `custom_params["__req__"] = self` reference with `weakref.proxy(self)` in `Req.__init__`, breaking reference cycles when custom logit processors (e.g., `Qwen3ThinkingBudgetLogitProcessor`) are attached and reclaiming memory immediately upon request completion/abort ($256\text{ MiB}$ reclaimed across 4096 aborted requests in our model).
2. **SGLang HiCache Skip Empty Auxiliary Transfers** (`sgl-project/sglang` PR `#45673`, commit `1596e88eda`):
   - Skips L2 transfer dispatch in `HybridCacheController._l2_transfers` when both `host_indices.numel() == 0` and `device_indices.numel() == 0`, avoiding zero-element transfer descriptors for idle `SWA` / `INDEXER` pools.
3. **SGLang Multimodal Precomputed Inputs Preserve Caller Hashes & Offsets** (`sgl-project/sglang` PR `#44160`, commit `ae579c5e00`):
   - Preserves caller-supplied `offsets` and media hashes for `PRECOMPUTED_EMBEDDING` items in `BaseMultimodalProcessor`, enabling RadixKV prefix reuse for pre-encoded multimodal embeddings.
4. **SGLang Diffusion Single-Pass VAE Tile Seam Blending** (`sgl-project/sglang` PR `#46240`, commit `d72664d903`):
   - Cross-fades adjacent VAE tiles in a single pass via `_blend(a, b, blend_extent, dim)`, cutting seam-blending read/write passes by $50\%$.
5. **vLLM Rust Frontend Prefiller Cache Hits for Disaggregated P/D Usage** (`vllm-project/vllm` PR `#58885`, commit `c36154742e`):
   - Propagates `remote_prefill_cached_tokens` through `GenerateOutputStream` in the Rust frontend so disaggregated P/D responses report true prefiller prefix cache hits instead of `0`.
6. **vLLM Finalize NIXL DCP Interleave Size Before Profiling** (`vllm-project/vllm` PR `#62330`, commit `3e75ccba29`):
   - Aligns NIXL `dcp_interleave_size` to `block_size=16` prior to memory profiling so KV cache planning matches runtime DCP interleaving.
7. **vLLM DeepSeek-V4 Hopper Block-FP8 Fallback & ROCm CDNA4 Default MXFP4 Indexer K Cache** (`vllm-project/vllm` PRs `#61875` `fcb787055e`, `#63669` `d0d681e40d`):
   - Declares MXFP8 MoE activations on Hopper (`SM90`) and executes them via block-FP8 (`group_size=128`); sets `mxfp4` ($0.53125\text{ B/elem}$, saving $46.88\%$ indexer K cache memory vs. FP8) as the default DeepSeek-V4 sparse indexer K cache dtype on AMD CDNA4 (`MI350/MI355`).
8. **TensorRT-LLM Named `PyTorchModelEngine` Init Phases & Default SM100 PrimTS Native Sparse MLA FMHA** (`NVIDIA/TensorRT-LLM` PRs `#20851` `011316470c`, `#20209` `178ac92fed`):
   - Splits `PyTorchModelEngine` initialization into named phases and enables `flashinfer-prims-ts` native sparse MLA FMHA by default for DeepSeek-V4 generation on SM100 ($16\times$ KV read reduction at $L=32768, \text{top\_k}=2048$).

## 2. Architect Exercises & Falsifiable Hypotheses

- **Sizing Exercise**: Switching the DeepSeek-V4 indexer K cache from FP8 ($1.0\text{ B}$) to MXFP4 ($0.5 + 1/32 = 0.53125\text{ B}$) reduces indexer K cache footprint by $46.88\%$.
- **Falsifiable Hypothesis**: Reporting only decode-local `cached_tokens` in disaggregated P/D serving under-reports prefix cache hit ratios to zero whenever prefill executes on a remote worker.
