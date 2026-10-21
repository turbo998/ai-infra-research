# SGLang Mooncake Failed-Session Exponential Probe & NIXL UCX Binding, GLM-5.2 Side-Stream Join, and TensorRT-LLM GLM-5.3 `glm_kpool` Sparse Attention

> **First-Hand Upstream Evidence (2026-10-21)**:
> - `sgl-project/sglang` PR `#45338` (`be65391abd01f711cdc443a3fe8f508b0c960357`): `[PD] Default-enable Mooncake failed-session probe with backoff and idle pruning`
> - `sgl-project/sglang` PR `#43482` (`2e7c9d851c868678ddfcd93b4e8000cd23e1f779`): `[PD] Support per-GPU UCX device selection in NIXL disaggregation`
> - `sgl-project/sglang` PR `#45349` (`f7b8c6fe404ca4f29c3188cb50692f7735d45a99`): `[GLM-5.2] Record DSA side-stream work right before its join`
> - `sgl-project/sglang` PR `#44929` (`da7e3f1c125be0f1ef2101c474b679296acdb325`): `[UMBP] Accept packed draft layer mappings in UMBPDirectLinker`
> - `NVIDIA/TensorRT-LLM` PR `#20454` (`ef913d39b3b5ba488dc0972bd400e52b41899767`): `[https://nvbugs/6068971][perf] Speed up glm_kpool sparse attention for GLM-5.3-Flash`
> - `NVIDIA/TensorRT-LLM` PR `#19953` (`4636b6311f55ec4ffe16d44fba3601f41638ec42`): `[https://nvbugs/6039511][feat] Enable CUTLASS MXFP8 MoE on SM107`

---

## 1. SGLang Mooncake Failed-Session Probe with Exponential Backoff & NIXL UCX Per-GPU Binding (PR `#45338` / `#43482`)

1. **Default-On Mooncake Failed-Session Exponential Backoff & Idle Pruning (PR `#45338`)**:
   - Enables `SGLANG_ENABLE_FAILED_SESSION_PROBE = True` by default in `environ.py` and tracks blacklisted decode peers via `FailedSessionProbe(last_requested, next_probe_at, interval)` in `mooncake/conn.py`.
   - Failed probes double `interval` from `30.0s` up to `SGLANG_FAILED_SESSION_PROBE_MAX_INTERVAL_S = 300.0s` (`t = 30s, 90s, 210s, 450s`), and sessions unrequested for `SGLANG_FAILED_SESSION_IDLE_TIMEOUT_S = 600.0s` are pruned at `t = 750s`—cutting 24-hour probe traffic for a dead decode session from **`2880` probes down to `4` (`-99.86%`)**.
   - Validates probe interval invariants in `__init__` across all roles so a later decode-to-prefill role switch cannot fail dynamically.
2. **Per-GPU UCX HCA Selection in NIXL Disaggregation (PR `#43482`)**:
   - Adds `_resolve_ucx_device_list(ib_device, gpu_id)` in `nixl/conn.py` using `get_ib_devices_for_gpu`, rejecting `:port` suffixes (since NIXL UCX appends port `1`) and joining HCA names with the literal `", "` delimiter required by NIXL's UCX `device_list` parser.

---

## 2. GLM-5.2 DSA Side-Stream Join Ordering & UMBP Packed Draft Layers (PR `#45349` / `#44929`)

1. **Tightly Bracketed Side-Stream Join for GLM-5.2 CUDA Graphs (PR `#45349`)**:
   - Moves `with torch.cuda.stream(self.alt_stream):` (`wq_b` in `dsa_indexer.py` and `q_b_proj_forward` in `forward_mla.py`) immediately after the main-stream projection/indexer and right before `current_stream.wait_stream(self.alt_stream)`, keeping the CUDA-graph replay join strictly on the main stream lane.
2. **Packed Speculative Draft Layer Mappings in UMBP Direct Linker (PR `#44929`)**:
   - Extends `_ordered_layers` and `_range_items` in `umbp_direct_linker.py` to accept `int | Sequence[int]` per logical layer (e.g., `{60: [0, 1], 61: 2}`), validating a contiguous bijection across all buffer indices while deduplicating logical layer order via `dict.fromkeys`.

---

## 3. TensorRT-LLM GLM-5.3-Flash `glm_kpool` Sparse Attention Speedups & Rubin `SM107` MXFP8 MoE (PR `#20454` / `#19953`)

1. **Single-Launch Fused Decode State Append (`_append_paged_state_kernel`)**:
   - Replaces ~`6` separate PyTorch indexing/quantization/scatter launches on the decode path (`request_ids is None`) with a single Triton kernel (`append_paged_state_rows`) that computes paged slots, scales/clamps FP8 E4M3 latent rows, and writes both `latent_pool` and `index_pool`.
2. **2D Draft-Verification Same-Pool Skip (`_kpool_update_kernel` `ROWS_PER_TABLE > 1`)**:
   - Accepts 2D `positions[N, T]` during speculative verification and skips rows whose successor in the same table belongs to the same `KPOOL = 64` block (`POS[b + 1] // KPOOL == pos // KPOOL`), ensuring each touched pool is refreshed only once by its final visible member (`8 -> 3` updates across two 4-token verification trees).
3. **Native TRTLLM-GEN FP8 Sparse Decode & Pre-Multiplied Host `bmm1_scale`**:
   - Extends `can_run_native` to serve FP8 E4M3 KV caches up to `128` query heads (`head_dim == 512, tokens_per_block == 64, topk % 64 == 0`), avoiding the `[tokens, topk, 512]` BF16 gather/dequantize fallback, and precomputes `_native_bmm1_scale = softmax_scale * kv_scale_quant_orig` on the host outside CUDA Graph capture.
4. **Rubin `SM107` CUTLASS MXFP8 MoE (PR `#19953`)**:
   - Adds `107` to `CutlassFusedMoE` `QuantAlgo.MXFP8` `sm_constraint` (`{100, 103, 107}`).

---

## 4. Deterministic Verification

```bash
python experiments/sglang_mooncake_probe_glm52_trtllm_glm53_sparse/run.py --check
```
