# Phase 91 Track C: FlashAttention-4 SM100 Sparse MLA & ROCm/CK Split-KV, Megatron-LM Sequence Packing, DeepEP Proxy Fence, verl CP Distillation, Mooncake SSD Prefetch & Weight Groups, NIXL NVTX Correlation & Dynamo KV-Router

Deterministic Python 3.12 standard-library model covering 10 first-hand commits:

1. `Dao-AILab/flash-attention@2cb0a8e471`: Guarding padded Q-head rows (`real_q_heads < 128`) in SM100 CuTe sparse-MLA `o_lo` residual register-to-global stores with `is_valid_qhead_row`.
2. `Dao-AILab/flash-attention#2901` (`0b4cb4e64b`): Fixing ROCm/CK `override_num_splits_if_necessary` (`seqlen_k` block count instead of constant `1`, `nhead_k` instead of `nhead` for GQA, and capping `num_splits <= 8` for CK combine numerical stability, yielding up to `3.21x` speedup).
3. `NVIDIA/Megatron-LM#8378` (`ede2c586cf`): Returning `False` in `is_dataset_built_on_rank` on middle pipeline stages when `sequence_packing_scheduler` is enabled.
4. `deepseek-ai/DeepEP#642` (`d4f41e4e93`): Inserting `fence.proxy.async.shared::cta` (`fence_view_async_shared()`) before `mbarrier_arrive` in `internode_ll.cu` low-latency combine receive.
5. `volcengine/verl#8332` (`67bf1df3f2`): Packing distillation teacher top-k tensors with the student's `cp_layout` (`dsv4_hybrid` contiguous vs `zigzag`) and `local_cp_size` in `compute_forward_kl_topk`.
6. `kvcache-ai/Mooncake#4272` (`f751cff2b6`): Read-only replica queries (`QueryReadOnly`, zero lease grant) and non-blocking SSD-to-DRAM promotion on `is_exist` (`prefetch_to_memory = true`).
7. `kvcache-ai/Mooncake#4313` (`2e533c97fe`): Atomic HA-replicated managed weight group revision lifecycle and protection from generic cache eviction.
8. `ai-dynamo/nixl#2495` (`a95227d202`): Carrying `nixl.xfer_request_id` as the `uint64` NVTX payload on both sender (`postXferReq`, `nixl::xfer.complete`) and receiver (`nixl::notif.received`).
9. `ai-dynamo/dynamo#16765` (`75d0686fb7`): Moving `sglang_response` directly into `Value::Object(engine_data)` in `native_http.rs` to eliminate host copies of large `routed_experts` payloads.
10. `ai-dynamo/dynamo#17149` (`30e972003a`): Scalar lower-tier KV-router residency walk (`OwnerView::exact_id`) for single-worker + single-hint-source queries.
