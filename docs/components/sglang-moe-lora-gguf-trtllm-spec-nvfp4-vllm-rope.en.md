# SGLang MoE LoRA Virtual-Expert Routing & GGUF Subnormal Weighting, TensorRT-LLM DeepSeek-V4 NVFP4 KV-Tile Padding & DeviceCopyBuffer Prefix Guard, and vLLM Muse Spark Shared FP32 RoPE & DeepSeek-VL2 Cropping Cache Identity

> **中文版**: [sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope.md](sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope.md)  
> **Phase 63 · Priority 72 · Track B (`question-203`)** · Verified Date: `2026-10-24` (UTC+08:00)

---

## 1. Core Architectural Problem & Context

Across multi-tenant MoE LoRA serving, Blackwell NVFP4 sparse MLA prefill, multi-step speculative decoding, and multimodal prefix caching, SGLang, TensorRT-LLM, and vLLM resolved six subtle correctness and memory bottlenecks:

1. **SGLang MoE LoRA Virtual-Expert Out-of-Bounds & Disabled-Slot Bleed-Through**: In `moe_lora_align_block_size` and `_fused_virtual_topk_ids_kernel`, when a request's `lora_id` targeted a slot disabled for a specific projection (`adapter_enabled[lora_id] == 0`) or exceeded `num_slots`, previous logic failed to emit `-1` and on the `SHARED_OUTER` shared-expert path clamped invalid `lora_id` values to slot `0`, corrupting outputs with slot `0`'s adapter weights.
2. **SGLang GGUF MoE Fused Weighting Negative-ID Out-of-Bounds & SM90+ FTZ Drift**: Masked top-$k$ slots (`ids[token, slot] < 0`) were not zeroed in-place, and standard floating-point multiplication on SM90+ flushed subnormal products to zero under `-ftz=true`, diverging from `torch.mul`.
3. **TensorRT-LLM DeepSeek-V4 NVFP4 Sparse MLA FMHA Out-of-Bounds Tile Reads**: `nvfp4_compress_fp8_scratch` was sized to exact `num_rows`, whereas `sparse_mla_fmha` loads `_SPARSE_MLA_FMHA_KV_TILE_SIZE = 128` rows per tile, reading past buffer end when segment lengths were not multiples of `128`.
4. **TensorRT-LLM Speculative Decoding Redundant H2D Copies & Tail Overwrites**: Unconditionally copying host metadata into CUDA-graph buffers on every draft step incurred avoidable H2D overhead and overwrote trailing slots when copying shorter prefixes.
5. **vLLM ROCm Muse Spark Redundant Per-Layer FP32 RoPE Tables**: Under speculative decoding, every non-draft target layer instantiated its own FP32 `cos_sin_cache` (`64 MiB` per layer at $128\text{k}$ context, wasting **`2.5 GiB`** across 40 layers).
6. **vLLM DeepSeek-VL2 Dynamic-Cropping Multimodal Prefix Cache Collision**: DeepSeek-VL2 enables local image cropping only when `len(images) <= 2` and disables cropping when `len(images) >= 3`. Because `_get_mm_hashes` omitted `cropping = len(images) <= 2`, an image cached in a 2-image request collided with the same image in a 3-image request.

---

## 2. Upstream Implementation Paths & Quantitative Mechanics

### 2.1 SGLang: MoE LoRA Virtual-Expert Contract (`d0022e1df3` / `#45681`) & GGUF `__fmul_rn` Fidelity (`e6451c1a95` / `#45916`)

1. **Strict Virtual-Expert Gating (`compute_virtual_id`)**: In `sgl-kernel/csrc/lora/moe_lora_align_kernel.cu`, commit `d0022e1df3` introduces `adapter_is_enabled<HasAdapterEnabled>(lora_id, adapter_enabled, num_slots, enabled_stride)`:
   $$\text{virtual\_id}(e, l) = \begin{cases} -1, & \text{if } l < 0 \lor l \ge S \lor \neg\text{enabled}[l] \lor e < 0 \\ l, & \text{else if } \text{SHARED\_OUTER} \\ -1, & \text{else if } e \ge E_{\text{weight}} \\ e + l \cdot E_{\text{weight}}, & \text{otherwise} \end{cases}$$
   It also adds `return_inactive_prefix_count=True` to return `[total_padded_tokens, leading_negative_expert_blocks]` so downstream SGMM/Cutlass kernels skip inactive `-1` prefix blocks, and validates `0 <= slot < MAX_LORAS` and `0 <= group < GROUPS_PER_SLOT` in `route_bucket_ids` before casting to `int32`.
2. **GGUF MoE Non-FTZ Weighting (`apply_gguf_moe_weights_kernel`)**: Commit `e6451c1a95` zeroes output rows whenever `expert_id < 0` and uses `__fmul_rn(value, weight)` on `__CUDA_ARCH__ >= 900` to preserve IEEE-754 subnormal products.
3. **Polymorphic Radix Cache Actions (`005333d029` / `#45444`) & Quark FP8 Fallback (`9d7ad308bd` / `#45909`)**: Rust `sglang-radix-tree` replaces monolithic `CacheAction` enums with `Box<dyn CacheAction>` trait objects (`requires_insert_barrier()`), while `QuarkW8A8Fp8MoEMethod` falls back cleanly when inputs are non-FP8 or non-contiguous on ROCm.

### 2.2 TensorRT-LLM: DSv4 NVFP4 KV-Tile Padding (`36d8aef55e` / `#20119`) & `DeviceCopyBuffer` (`dd63b36fcf` / `#19989`)

1. **NVFP4 Sparse MLA FMHA Tail Padding**: Commit `36d8aef55e` defines `_SPARSE_MLA_FMHA_KV_TILE_SIZE = 128` and allocates `(num_rows + 128, head_dim)` zero-initialized scratch in `nvfp4_compress_fp8_scratch` (only `73,728 B` = `72 KiB` tail padding for `head_dim = 576`), preventing out-of-bounds tile reads at segment boundaries.
2. **`DeviceCopyBuffer` Prefix Skip & Kimi K3 Memory Model**: Commit `dd63b36fcf` introduces `DeviceCopyBuffer`, which compares `torch.equal(self._last_host[:n], flat)` to skip redundant H2D copies while preserving trailing elements `[n:]` on shorter updates (`force=True` overrides when C++ `MambaHybridCacheManager` mutates buffers in-place). The commit also documents the Kimi K3 (`1T` MoE, `960 GB` weights) per-rank memory budget on 2x B200 nodes: **`210 GB/rank`** under `DEP16` (`114 GB` replicated BF16 non-expert + `90 GB` MXFP4 EP16) vs **`115 GB/rank`** under `TEP16` (leaving **~75 GB/rank** for KV cache).

### 2.3 vLLM: Muse Spark Shared FP32 RoPE (`b1c66e5545` / `#62757`), CPU Pooling KV Skip (`c3117c7441` / `#59405`), & DeepSeek-VL2 Hash (`23f546eee0` / `#63120`)

1. **Muse Spark Shared FP32 `cos_sin_cache`**: Commit `b1c66e5545` shares a single `get_rope` instance across all $L$ non-draft target layers in `MuseSparkTextModel`. For $L=40, T_{\max}=131,072, d_{\text{head}}=128$ (`float32`, `64 MiB` per table), sharing reclaims **`2,496 MiB` (`2.4375 GiB`, $40\times$ reduction)**.
2. **CPU Pooling Runner KV Check Bypass**: Commit `c3117c7441` skips the `available_cpu_memory < total_kv_cache_bytes` check in `CPUWorker` when `runner_type == "pooling"`.
3. **DeepSeek-VL2 `cropping` Hash Isolation**: Commit `23f546eee0` passes `cropping = len(images) <= 2` into `MultiModalHasher.hash_kwargs` so cropped ($\le 2$ images) and uncropped ($\ge 3$ images) visual token sequences never collide in the prefix cache.

---

## 3. Deterministic Verification & Architect Guidance

| Verification Sub-Test (`experiments/sglang-moe-lora-gguf-trtllm-spec-nvfp4-vllm-rope/run.py`) | Key Parameters | Deterministic Result | Architect Takeaway |
|---|---|---|---|
| SGLang MoE LoRA `compute_virtual_id` (`d0022e1df3`) | `num_slots=4, num_experts=8, enabled=[1,0,1,1]` | Disabled slot `1` $\to -1$; OOB slot `4` $\to -1$; active `(e=5, l=2)` $\to 21$ | Use `return_inactive_prefix_count` to skip leading `-1` blocks; never clamp invalid slots to `0` |
| TRT-LLM NVFP4 FMHA Scratch Padding (`36d8aef55e`) | `num_rows=1024, head_dim=576, tile=128` | Tail zero pad `73,728 B` (`72 KiB`), total `663,552 B` | Always pad compressed KV scratch by one full FMHA KV tile (`128` rows) |
| TRT-LLM `DeviceCopyBuffer` (`dd63b36fcf`) | 5-step speculative update trace | `3` copies enqueued, `2` skipped, tail `[30, 40]` preserved | Guard CUDA-graph H2D metadata copies with prefix equality; pass `force=True` after C++ in-place writes |
| vLLM Muse Spark Shared RoPE (`b1c66e5545`) | $L=40, T_{\max}=131072, d=128$ (`float32`) | `64.0 MiB` single table, **`2,496.0 MiB` saved** ($40\times$) | Share identical FP32 RoPE caches across homogeneous target layers |
| vLLM DeepSeek-VL2 Cropping Hash (`23f546eee0`) | Same image at $N=2$ vs $N=3$ images | Distinct SHA-256 cache digests (`collision_prevented = True`) | Include batch-dependent preprocessing flags (`cropping`) in multimodal cache keys |
