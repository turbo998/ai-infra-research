# Phase 111 Track B: SGLang DFlash SpecForge MLA & Async NaN Probe, vLLM RDNA/MORI-IO/Overlaid KV Copy, and TensorRT-LLM Helix CP Consensus & MiniMax-H3 Fused RoPE

> **Provenance & Evidence Boundary**: Derived from first-party public GitHub commits across SGLang, vLLM, and TensorRT-LLM plus our deterministic stdlib model (`experiments/sglang_dflash_mla_vllm_moriio_trtllm_helix_cp_phase111/run.py`).

---

## 1. SGLang: DFlash SpecForge MLA Draft Serving, Fused Async NaN Probe, NUMA Physical Core Budget & DiT BCG Outputs

### 1.1 DFlash Serving SpecForge MLA Drafts via Target MLA Attention (`sgl-project/sglang@b068417bfcc1`, `#44676`)
- **Implementation Path**:
  - Publishes target MLA attention backend for SpecForge MLA drafts in `speculative_hook.py` (`_publish_dspark_mla_draft_backend`, `_keep_mla_draft_off_trtllm_mha`).
  - Adds `set_mla_kv_buffer_prefix_valid` in `memory_pool.py`: redirects uncommitted draft slots past `commit_lens[:, None]` to reserved physical slot `0` so MLA KV writes skip them cleanly.

### 1.2 Single-Kernel Fused Async NaN Validation (`ee5443f53026`, `#43720`), NUMA Budget (`302d5e7b8ca9`, `#45516`) & DiT BCG Outputs (`c43abd6d9e38`, `#46659`)
- **Fused Async NaN Probe (`ee5443f53026`)**: Replaces `torch.isnan(x) -> .any() -> torch._assert_async()` (3 kernel launches + intermediate boolean tensor) with one Triton kernel `_assert_no_nan_kernel` (`@triton.jit(debug=True)`, `_NAN_CHECK_BLOCK = 1024`).
- **NUMA Empty-Node Budget Fix (`302d5e7b8ca9`)**: Caches `get_cpu_ids_by_node()` with `@lru_cache(maxsize=1)` to prevent CPU memory budget overestimation on hosts with empty NUMA nodes.
- **DiT Attention BCG Pre-Allocated Buffers (`c43abd6d9e38`)**: Allocates output tensors before breakable CUDA graph (`bcg`) break points so graph replay reads refreshed attention outputs.

---

## 2. vLLM: ROCm RDNA 2D PagedAttention, SM90 GDN Tile, MORI IO Strided Block Length & Overlaid KV View Copy

### 2.1 ROCm RDNA `kernel_paged_attention_2d` (`9af236a33281`, `#63735`) & SM90 GDN Tile (`94f56ce92cc4`, `#63864`)
- **ROCm RDNA Launch Config (`9af236a33281`)**: Forces `num_stages=1` and scales `num_warps` to `8` (`tile_size >= 4096`) or `16` (`tile_size >= 8192`) on RDNA (`gfx12`), preventing GPU memory faults from 2-stage Triton buffer pipelining.
- **SM90 Packed GDN Decode (`94f56ce92cc4`)**: Selects `BV = 4` on SM90 when `K == V == 128` and `initial_state.dtype == torch.float32` while retaining `BV = 32` for BF16 states.

### 2.2 DeepSeek-V4-Pro MORI IO PD Disaggregation (`3d81a510f9eb`, `#48989`) & Overlaid KV Block Copy (`8701c9b7513d`, `#55112`)
- **MORI IO Stride-Aware `block_len` (`3d81a510f9eb`)**: Computes `block_len = (1 + sum((s - 1) * st for s, st in zip(shape[1:], stride[1:]))) * element_size` in `moriio_layout.py`, covering padded inner strides (`38,880 B` vs `36,864 B` naive product).
- **Overlaid KV View Copy (`8701c9b7513d`)**: Keys `seen` in `copy_kv_cache_blocks_inplace` by `(device, data_ptr, tuple(shape), tuple(stride), element_size)` so a narrow Mamba state view (`24 B/block`) sharing `data_ptr` with a full attention KV view (`64 B/block`) no longer suppresses copying the remaining `40 B` per block.

---

## 3. TensorRT-LLM: Helix CP + Attention-DP Gen Transfer Consensus (`94348c4de6eb`, `#19575`) & MiniMax-H3 Fused QK-Norm+RoPE (`af32051e1926`, `#19645`)

- **Helix CP Admission Consensus (`94348c4de6eb`)**: `DisaggTransferCoordinator.align_gen_admission_across_cp` intersects admitted `gen_init` requests across Helix CP ranks and requires identical DP target routing before entering the gen transfer status collective, returning dropped requests' V2 KV blocks for retry.
- **MiniMax-H3 Fused QK-Norm + Partial Split-Half RoPE (`af32051e1926`)**: Fuses Q and K per-head RMSNorm and leading split-half RoPE into a single Triton launch (`1 token x 8 heads/program`, `1 warp`), cutting BF16 roundings from `5` to `1` (`rel. L2 = 3.5e-3`).
