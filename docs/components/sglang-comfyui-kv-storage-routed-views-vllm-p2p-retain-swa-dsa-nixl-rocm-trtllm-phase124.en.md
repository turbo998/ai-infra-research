# SGLang ComfyUI SGLDiffusion, StaticKVStorage & Routed Draft KV Views, vLLM P2P Retained Source Pins, SWA Lazy Offload & DSA Decode Indexer Memory, NIXL Push, ROCm MXFP8 Fusion & TensorRT-LLM 64-Bit AttnRes Stride Deep Dive (Phase 124)

> **中文版**: [sglang-comfyui-kv-storage-routed-views-vllm-p2p-retain-swa-dsa-nixl-rocm-trtllm-phase124.md](sglang-comfyui-kv-storage-routed-views-vllm-p2p-retain-swa-dsa-nixl-rocm-trtllm-phase124.md)
> **Evidence Tier**: `implementation` + `author-benchmark` + `own-offline` (Zero GPU hardware re-measurement claims)

---

## 1. Architectural Context & Problem Statement

Across multimodal diffusion serving, disaggregated prefill/decode (PD), P2P RDMA KV offloading, and long-context sparse attention engines, Phase 124 Track B resolves eleven critical correctness and latency bottlenecks:

1. **Silent Kwarg Dropping in ComfyUI SGLDiffusion Adapters (`4517a5ff1b`, `b1c6945659`)**: Unsupported conditioning kwargs (`attention_mask`, `additional_t_cond`, `siglip_feats`, `ref_contexts`) were silently dropped at the ComfyUI adapter boundary, and users lacked visibility into whether `--disable-cuda-malloc` zero-copy CUDA IPC was active.
2. **Coupled KV Byte Storage and Allocation Identity Spaces in SGLang (`146378629d`, `53592ea5ff`, `ca8184e13a`)**: Beam-search fork, disaggregated prefill/decode, and host KV offload bypassed `KVCache` contracts and conflated byte movement with `IdSpaceKind` (`TOKEN_SLOT`, `PAGE_BLOCK`, `MHA_WINDOW_SLOT`, `LINEAR_STATE_SLOT`).
3. **Missing Routed KV Views in Speculative Drafts, In-Flight RDMA Compaction Hazards & ROCm Translated Window Writes (`240862923e`, `eb79ecca11`, `14fea75124`)**: Fused speculative drafts did not bind `get_routed_kv_views`; disaggregated RDMA transfers needed allocator-boundary `disagg_transfers_in_flight` pin tracking to block compaction; ROCm enabled `enable_fused_set_kv_buffer` on window-translated pools.
4. **DMA Source Buffer Use-After-Free on vLLM P2P KV Offload Teardown (`a1cfd2c1b5`)**: `remove_remote_peer()` and `session.close()` unpinned source KV blocks immediately after requesting cancellation of in-flight `WRITE` transfers while the NIC DMA engine was still reading them.
5. **Underestimated SWA Lazy Offload Target Blocks & Unreserved DSA Decode Indexer Logits (`77f58e5304`, `317081fb6d`)**: `_estimate_lazy_target_blocks` iterated only over `prefix_cacheable_groups` (omitting `SlidingWindowMLASpec`), and DSA `sparse_attn_indexer` skipped decode indexer logits during memory profiling (`64.0 MiB` FP32 at `256×65536`).
6. **NIXL Push Mode, ROCm MiniMax-M3.1 MXFP8 Norm Fusion & SpecDecode Adaptive Padding (`066d979934`, `1cb31be702`, `5118d990fb`)**: `NixlConnector` needed automatic `transfer_mode: "push"` dispatch to `NixlPushConnector`, and ROCm MiniMax-M3.1 fused MXFP8 E4M3 quantization into Gemma `input_layernorm`.
7. **TensorRT-LLM Kimi-K3 32-Bit Signed Overflow in `attn_res`, Eager GDN `torch.cat` Overhead & GLM-5.3-Flash KPool (`e2b1544160`, `99466ea9ad`, `2fe91cdfc9`)**: In `attnResBlock`, `int block_stride_r = num_tokens * hidden_dim` overflowed 32-bit `INT_MAX` (`2,147,483,647`) when `source * block_stride_r = 7 * T * 7168` reached **`T >= 42,800`** (`2,147,532,800 > INT_MAX`).

---

## 2. Implementation Mechanics & Quantitative Evidence

### 2.1 SGLang ComfyUI SGLDiffusion & `StaticKVStorage` Unification
- **Fail-Fast Conditioning & CUDA IPC (`4517a5ff1b`, `b1c6945659`)**: Explicitly rejects unsupported conditioning kwargs and adds `SGLDWorkerStatus`; with `--disable-cuda-malloc` zero-copy CUDA IPC on H100, Flux + ControlNet (`1024×1024`, 25 steps) drops from `13.1 s` to **`4.7 s` (`2.7872x` speedup)** and Wan2.2 TI2V 5B drops from `132.0 s` to **`39.4 s` (`3.3503x` speedup)**.
- **`StaticKVStorage`, Routed Draft Views & Disagg Pins (`146378629d`..`14fea75124`)**: Unifies row/token copying under `kv_storage_of(pool)` while separating `IdSpaceKind`, binds fused speculative drafts to `get_routed_kv_views`, blocks compaction whenever `disagg_transfers_in_flight > 0`, and gates ROCm `enable_fused_set_kv_buffer` on `not _window_writes_translated(pool)`.

### 2.2 vLLM P2P Retained Source Pins, SWA Offload & DSA Indexer Memory Profile
- **Retained Source Pins (`a1cfd2c1b5`)**: `remove_remote_peer()` and `session.close()` invoke `cancel(mode="wait")` on in-flight `WRITE` transfers and move them to `_retained` (holding source pins up to `_RETAINED_HOLD_CAP_S = 30.0 s` until `poll_retired()` confirms completion), eliminating DMA source UAF and reconnect peer-ID collisions.
- **SWA Lazy Target Blocks (`77f58e5304`) & DSA Decode Logits (`317081fb6d`)**: Iterates all `kv_cache_groups` (including `SlidingWindowMLASpec`, fixing a `1.3333x` block ratio on 48-full + 16-SWA models) and reserves `[num_reqs, max_seq_len]` decode indexer logits (`64.0 MiB` FP32 at `256×65536`) during `is_profiling`.
- **NIXL Push & ROCm MXFP8 (`066d979934`, `1cb31be702`, `5118d990fb`)**: Routes `transfer_mode: "push"` to `NixlPushConnector`, fuses MXFP8 E4M3 quantization into Gemma `input_layernorm` via `MinimaxM3RMSNormQuantFusionPass`, and passes `max_query_len=None` during adaptive draft prefill padding.

### 2.3 TensorRT-LLM Kimi-K3 64-Bit Stride, Eager GDN Pre-Allocation & GLM-5.3-Flash KPool
- **64-Bit `block_stride_r` (`e2b1544160`)**: Promotes `block_stride_r` to `long long` (`int64_t`), preventing signed 32-bit wrap-around (`2,147,532,800 -> -2,147,434,496`) at $T \ge 42,800$ for Kimi-K3 ($H = 7168, N = 9$).
- **Eager GDN Pre-Allocation (`99466ea9ad`) & GLM-5.3-Flash KPool (`2fe91cdfc9`)**: Pre-allocates a single `(1, T, num_v_heads_per_tp, head_v_dim)` output tensor to eliminate `torch.cat` in mixed prefill+decode steps, and accelerates GLM-5.3-Flash concurrency-1 output speed by **`+6.0%`** on 4×B300 (`EP=1, MTP=3, FP8 KV`).

---

## 3. Deterministic Verification

- **Verification Script**: `experiments/sglang_comfyui_kv_storage_routed_views_vllm_p2p_retain_swa_dsa_nixl_rocm_trtllm_phase124/run.py`
