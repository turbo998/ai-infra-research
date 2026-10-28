# Phase 128 Deep Dive (Track C): TensorRT-LLM Helix CP Pre-Exchange V Up-Projection & Eager GDN, Megatron-LM GDP FP32 Sigmoid & Batch Transition, PyTorch Inductor Cat Fusion & 64-Bit Pool3D, and Mooncake/Dynamo Contracts

## 1. Motivation & Core Problem

Across context-parallel MLA decoding (TensorRT-LLM Helix CP), Gated Delta Product (GDP) state-space training (Megatron-LM), compiler lowering (PyTorch Inductor), and distributed KV runtimes (Mooncake TENT and NVIDIA Dynamo), subtle cross-proxy ordering gaps, BF16 saturation rounding, 32-bit slice indexing overflows, and ambient environment-variable probing trigger data corruption, vanishing gradients, illegal memory accesses, or port/connection leaks. This report analyzes 10 upstream commits merged on 2026-10-28 across `NVIDIA/TensorRT-LLM`, `NVIDIA/Megatron-LM`, `pytorch/pytorch`, `kvcache-ai/Mooncake`, and `ai-dynamo/dynamo`.

## 2. Key Implementation Paths & Architectural Mechanisms

### 2.1 TensorRT-LLM Helix CP Pre-Exchange V Up-Projection, FIFO Async Fence, Eager GDN Shared Buffer & Beam Unit-Temp Skip (`b613fbe387ed`, `99466ea9ad97`, `3b4eebcf020e`)
1. **Helix CP MLA Pre-Exchange V Up-Projection & FIFO Async Fence (`b613fbe387ed`)**:
   - Registers `helix_v_b_proj_full` (`[num_heads_tp, v_head_dim, kv_lora_rank]`) so generation can project partial MLA outputs from `kv_lora_rank = 512` to `num_heads_tp * v_head_dim` before the CP All-to-All exchange (at `TP=64` with 2 local heads of dim 128, `256` vs `512` floats per token — a **`2.0x` All-to-All payload reduction**), while `Request.get_input_token_ids_array()` exposes a single-`memcpy` NumPy `int32` array for `O(prompt/cp)` partitioning.
   - Inserts `fence_view_async_shared()` before `__syncwarp()` in `helixAllToAllKernel` so next-iteration `cp.async.bulk` async-proxy refills are strictly ordered after generic-proxy `sanitize/protoPack` stores and S2G loads.
2. **Eager GDN Shared Output Buffer (`99466ea9ad97`) & Beam Search Unit-Temperature Skip (`3b4eebcf020e`)**: Pre-allocates a single `attn_out` buffer in eager mixed batches so prefill and decode write directly into disjoint slices (`0` concatenation copies), and leaves `temperature = None` when all beam-search requests have `temperature == 1.0` to skip full-vocabulary logit rescaling.

### 2.2 Megatron-LM GDP `beta` FP32 Sigmoid Gradient Preservation & Global-Batch Transition Detection (`a296523510e1`, `d1eec51b8563`)
1. **GDP `beta` FP32 Sigmoid (`a296523510e1`)**: Changes `b.sigmoid()` to `b.float().sigmoid().to(b.dtype)` in `GatedDeltaProductMixer._compute_gating`. In BF16 (step `2^-7 = 0.0078125` near `1.0`), `sigmoid(8.5) = 0.9997966` rounds to `1.0`, collapsing the backward derivative `sigmoid * (1 - sigmoid)` to **`0.0`**; computing the sigmoid in FP32 preserves the exact non-zero gradient **`0.00020339`**.
2. **Global-Batch Transition Detection (`d1eec51b8563`)**: Adds `is_batch_size_transition` comparing both `get_num_microbatches()` and `get_current_global_batch_size()`. When a step schedule increases global batch size from `8,192 -> 12,288` at `DP = 8,192` (`mbs = 1`), integer floor division leaves `num_microbatches = 1` unchanged; checking `global_batch_size` catches the transition deterministically.

### 2.3 PyTorch Inductor `pointwise_cat` Internal-Consumer Exclusion & CUDA `max_pool3d` 64-Bit Slice Indexing (`dd50eb629ec0`, `53765f04b1b4`)
1. **Inductor `pointwise_cat` Internal Consumer Fix (`dd50eb629ec0`)**: For RoPE-style `cat((rotated, -rotated))`, `negated = -rotated` is also a direct input to the same `cat`. By excluding users in `cat_input_nodes = OrderedSet(input_nodes)` from external consumer counts, Inductor fuses `cat((rotated, -rotated))` via `pointwise_cat` instead of falling back to `ConcatKernel`.
2. **CUDA `max_pool3d` 64-Bit Slice Index (`53765f04b1b4`)**: Promotes `maxIndex` and loop `index` inside `max_pool3d_with_indices_single_out_frame` to `(int64_t(t) * iheight + h) * iwidth + w`, preventing 32-bit signed integer wrap-around (`+2,147,483,648 -> -2,147,483,648`) and illegal memory access when a single `T*H*W` spatial slice exceeds `2^31 - 1` elements.

### 2.4 Mooncake TENT Metrics Self-Observability Contract & Dynamo On-Demand NATS / Stream Release (`d3ef7b1c2821`, `1b6416a308ee`, `78ac1bc99e0a`)
- **Mooncake TENT Shared DSO Singleton (`d3ef7b1c2821`)**: Links `libtent_metrics.so` dynamically across `engine.so` and `store.so` so hybrid processes share **1 singleton** instead of spawning 2 conflicting singletons fighting over port `9100`, and returns HTTP `500` on Prometheus/JSON serialization errors.
- **Dynamo Plane-Gated NATS (`1b6416a308ee`) & vLLM Stream Release on Local Stop (`78ac1bc99e0a`)**: Connects to NATS only when `request_plane` or `event_plane` actually uses NATS (ignoring ambient `NATS_SERVER` pod env vars in TCP/ZMQ mode), and releases vLLM streams immediately upon local stop in `vllm_processor.py`.

## 3. Deterministic Verification Summary

| Component / Commit | Verified Property | Before | After | Key Impact |
|---|---|---:|---:|---|
| TRT-LLM Helix CP (`b613fbe387ed`) | `TP=64` MLA All-to-All payload / FIFO fence | `512` floats/token / un-fenced | `256` floats/token + `fence_view_async_shared` | **`2.0x`** payload cut + async-proxy safety |
| Megatron GDP (`a296523510e1`) | `b=8.5` BF16 sigmoid backward gradient | `0.0` (vanished at saturation) | `0.00020339` (FP32 sigmoid) | Preserves saturated gating gradients |
| Megatron Batch (`d1eec51b8563`) | `8192->12288` transition at `DP=8192` | `False` (`microbatches == 1`) | `True` (checks `global_batch_size`) | Eliminates missed schedule transitions |
| PyTorch Inductor (`dd50eb629ec0`) | `cat((rot, -rot))` external consumers | `1` (`ConcatKernel` fallback) | `0` (`pointwise_cat` fused) | Restores RoPE cat fusion |
| PyTorch Pool3D (`53765f04b1b4`) | `2048×1024×1024` slice offset | `-2,147,483,648` (int32 overflow) | `+2,147,483,648` (`int64_t`) | Eliminates >2G-element slice crash |
| Mooncake & Dynamo (`d3ef7b1c2821`, etc.) | TENT singletons / ambient NATS conns | `2` singletons / `1` bogus NATS conn | `1` shared singleton / `0` conns | Fixes port 9100 clash & stream leak |
