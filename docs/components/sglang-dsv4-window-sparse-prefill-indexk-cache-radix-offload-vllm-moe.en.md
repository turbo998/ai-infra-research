# SGLang DeepSeek-V4.1 WindowLayout & Per-Ratio Index-K Caching, Sparse Prefill Restoration Under Request Windows, RadixCache Dual-Stream Offload Ordering, and vLLM Monolithic MoE Routing Simulation Guards

> **First-Hand Upstream Sources**:
> - `sgl-project/sglang@f7d4f06242b76615a2cc95e87426b7123a90c7f6` (`[dsv4] Precompute WindowLayout history_src once per batch (#45579)`, 2026-10-21)
> - `sgl-project/sglang@bc412213c17cecfa826716e23d5ae00748bbe528` (`[dsv4] Cache V4.1 prefill index-k gather and plans per ratio (#45577)`, 2026-10-21)
> - `sgl-project/sglang@97945831f9ec29755099284243ab7d91336d314f` (`[dsv4] Restore sparse prefill when request window is active (#45573)`, 2026-10-21)
> - `sgl-project/sglang@620718c8ea67d8f1ef827f6be78a77b3a7830b41` (`[dsv4] Keep vision_topk off packed-ids under routing simulation (#45575)`, 2026-10-21)
> - `sgl-project/sglang@2c7a9e3b50828720951a5f7e71f42563d2df1b47` (`[RadixCache] Order cache linker offloads behind forward stream (#45559)`, 2026-10-21)
> - `vllm-project/vllm@b0e1f8052abb3de7c2fa8dc6234733104e829804` (`[MoE] Support simulated routing in monolithic MoE backends (#52780)`, 2026-10-21)

---

## 1. SGLang DeepSeek-V4.1 `WindowLayout` Precomputation, Per-Ratio Index-K Caching & Sparse Prefill Restoration

### 1.1 Batch-Once `WindowLayout.history_src` Precomputation (`sglang@f7d4f06242`)
Across all 60 layers of DeepSeek-V4.1 long-context window execution, `forward_prepare` previously repeated the exact same 2D fancy indexing `req_to_token[req_pool_indices[:, None], pos]` on every layer:
- Commit `f7d4f06242` adds `history_src: Optional[torch.Tensor] = None` to `WindowLayout`, gathering the base token-pool slots **once per batch** in `build_window_layout()` when `max_hist > 0` (`torch.where(valid_mask, ..., 0)`);
- Each layer then calls `_materialize_history_slots()` to apply only its layer-specific pad slot via `history_src.masked_fill(~window.valid_mask, layer_pad_slot)`, reducing 2D fancy-index launches by **60x (`60 -> 1`)**.

### 1.2 Caching `low_ratio_k_slots` and `FastIndexPlan` Per Compression Ratio (`sglang@bc412213c1`)
In DeepSeek-V4.1, 48 low-ratio layers share just two compression ratios (`r in {4, 8}`). Previously, every low-ratio layer gathered request history and extend tokens into a temporary buffer, strided by `::r`, divided by `r`, and rebuilt a `FastIndexPlan`:
- **Divisible Stride Equivalence**: Because request-window left bounds and history lengths `H` are exact multiples of `r` (`H % r == 0`), concatenating history positions (`0..H-1`) with strided extend positions (`ext[::r]`) and then taking `::r` is algebraically identical to taking `history[::r]` (`H // r` elements) followed by `ext[::r]`;
- Commit `bc412213c1` caches `low_ratio_k_slots[r]` and `low_ratio_index_plans[r]` on `DSV4Metadata`, cutting low-ratio prefill index-k gathers and plan builds by **24x (`48 -> 2`)**.

### 1.3 Restoring Sparse Prefill When Request Window Is Active (`sglang@97945831f9`)
A prior refactor gated `use_sparse_prefill` on `window_layout is None`, unintentionally forcing long-context extend batches with an active `request_window` (`window_layout is not None`, e.g., `total_seq_len = 8192 >= sparse_prefill_threshold = 4096`) onto dense `forward_prefill_ragged`:
- Commit `97945831f9` removes the `window_layout is None` restriction so long-context request-window batches execute the sparse `forward_prefill_context_stage` path.

---

## 2. SGLang RadixCache Dual-Stream Offload Ordering (`sglang@2c7a9e3b50`)

### 2.1 The Read-Before-Write KV Hazard Under Overlap Scheduling
When `UnifiedCacheLinker` (`MooncakeDirectLinker`, `UMBPDirectLinker`) runs with `offload_synchronize = False` (default async overlap), `offload()` issues D2H/RDMA copies on a dedicated `offload_stream`. Previously, `offload_stream` only waited on `torch.cuda.current_stream()`:
- Under overlap scheduling, cache operations (`evict`, `inc_lock_ref`) can be triggered on a control stream while attention KV writes are still executing on `forward_batch.forward_stream`;
- Waiting only on `current_stream` allowed `offload_stream` to copy KV pages before `forward_stream` finished writing them (`t = 12 us < 25 us`).

### 2.2 `record_offload_ready()` Dual-Stream Fence
Commit `2c7a9e3b50` attaches `forward_stream` to `UnifiedCacheLinker` and records an `_offload_ready_event` on `forward_stream` after each model forward pass. `_wait_for_offload_ready()` then orders `offload_stream` behind **both** `torch.cuda.current_stream()` and `_offload_ready_event` (`t >= 25 us`), eliminating silent KV corruption.

---

## 3. vLLM Monolithic MoE & SGLang DSV4.1 Routing Simulation Guards (`vllm@b0e1f8052a`, `sglang@620718c8ea`)

| Framework & Commit | Problem | Contract Fix |
| :--- | :--- | :--- |
| **vLLM Monolithic MoE Routing Simulation** (`vllm@b0e1f8052a`) | Monolithic MoE kernels (FlashInfer TRTLLM BF16 / FP8 / MXFP4) fuse routing inside the kernel and never invoke external Python `RoutingSimulator`. | Adds `router.kernel_routing_method_type` (returns `SimulatedNoop` when active), `router.supports_monolithic_kernels` (true when simulation is disabled, or when `DeepSeekV3` routing uses `moe_backend_LoadBalanced` synthetic bias via `prepare_monolithic_bias()`), and rejects monolithic backends for all other custom simulators so selection falls back to decomposed modular MoE. |
| **SGLang DSV4.1 `vision_topk` Packed-IDs Guard** (`sglang@620718c8ea`) | `DeepseekV4MoEGate` encodes `topk_indices` as `packed_ids` (`(expert_id << 16) \| token_idx`) for FlashInfer `cutlass_fused_moe`, which `FunctionOverrideRouter` does not rewrite under routing simulation. | Gates `enable_packed_ids` on `not RoutingSimulationConfig.is_enabled()`, keeping `vision_topk` on unpacked `topk_ids` whenever routing simulation is active. |

---

## 4. Deterministic Reproduction (`experiments/sglang-dsv4-window-sparse-prefill-indexk-cache-radix-offload-vllm-moe/run.py`)

```bash
python experiments/sglang-dsv4-window-sparse-prefill-indexk-cache-radix-offload-vllm-moe/run.py --check
```
