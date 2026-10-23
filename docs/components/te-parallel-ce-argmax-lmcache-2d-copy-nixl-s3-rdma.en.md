# TransformerEngine Fused Parallel Cross-Entropy Argmax, LMCache 2-D Block-Outermost Batch Copy & NIXL Multipart S3-over-RDMA

> **Language**: [中文](te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma.md) | [English](te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma.en.md)
> **Phase**: 65 (`priority: 74`) · **Track C**: Distributed Training, KV Cache Offloading & RDMA Object Storage
> **Verified Upstream Sources**: `NVIDIA/TransformerEngine@39a8b879df` (`#3671`), `NVIDIA/TransformerEngine@76ae4b0981` (`#3599`), `LMCache/LMCache@0d441be069` (`#5506`), `ai-dynamo/nixl@99d469be5d` (`#2406`), `ai-dynamo/nixl@14e329032b` (`#2172`), `kvcache-ai/Mooncake@d782a5998f` (`#4755`)

---

## 1. Core Architectural Problem & Background

In large-vocabulary tensor-parallel training, cross-process KV cache transfers on block-outermost memory layouts (such as DeepSeek-V4 `BLNHC`), and multi-gigabyte S3-over-RDMA object transfers, three recurring bottlenecks degrade throughput or correctness:

1. **Redundant Logits HBM Re-Reads & Non-NaN-Safe Triton `tl.max` in `parallel_cross_entropy` (`TransformerEngine@39a8b879df` & `76ae4b0981`)**:
   - Computing token-level accuracy alongside cross-entropy loss previously required calling `torch.argmax(logits)` after `parallel_cross_entropy`, reading the full vocabulary logits from HBM a second time even though the forward cross-entropy kernel had already scanned every row for its maximum.
   - Furthermore, Triton's `tl.max(..., return_indices=True)` is **not NaN-safe**: because comparisons with `NaN` evaluate to `False`, a block containing `[8, -10, NaN, -10]` returned index `1` (neither the maximum nor the `NaN`), violating `torch.argmax` semantics where `NaN` counts as the maximum and the first `NaN` (or first tie) wins.
   - Separately, in grouped MXFP8 quantization (`76ae4b0981`), the multi-tensor path (`is_single_tensor = false`) added `tensor_base / SCALE_DIM_X` to `scales_rowwise` while also setting `tensor_base_for_scales = tensor_base`, double-counting the scale offset.
2. **1-D Copy Descriptor Amplification in LMCache Block-Outermost Transfers (`LMCache@0d441be069`)**: In block-outermost KV layouts (e.g., vLLM's `BLNHC`), a single block's pages across all $N_L$ layers sit side-by-side at a constant byte pitch inside one allocation. Submitting one 1-D `cudaMemcpyBatchAsync` descriptor per `(kv_plane, layer, block)` generated $61 \times 32 = 1952$ descriptors for a 32-block chunk.
3. **`CU_OBJ_INVALID_VALUE` on $\ge 4\text{ GiB}$ S3-over-RDMA Transfers in NIXL (`nixl@99d469be5d` & `14e329032b`)**: Although CUDA 13.4 can register VRAM arenas well beyond 4 GiB (e.g., 64 GiB), `cuObject` issues an RDMA token (`cuMemObjGetRDMAToken`) for at most `CUOBJ_MAX_MEMORY_REG_SIZE` ($4\text{ GiB} - 64\text{ KiB} = 4,294,901,760\text{ B}$) per request. Issuing a single PUT or GET for an object $\ge 4\text{ GiB}$ failed with `CU_OBJ_INVALID_VALUE`.

---

## 2. Source-Level Mechanics & Mathematical Formulation

### 2.1 TransformerEngine NaN-Safe `parallel_cross_entropy(..., return_argmax=True)` (`39a8b879df`)

In `transformer_engine/common/triton/cross_entropy.py`, `_fold_block_argmax` maintains a separate `(run_max, run_idx, run_nan)` tuple alongside the online softmax statistics:
- Detects NaNs via `_is_nan(x)` (`x != x`), finds `first_nan = tl.min(tl.where(is_nan, offsets, n_cols))`, and masks NaNs to `-inf` before calling `tl.max(..., return_indices=True, return_indices_tie_break_left=True)`.
- If `run_nan == 0`: a block containing `NaN` sets `run_idx = first_nan, run_nan = 1`; otherwise `block_max > run_max` (strict inequality so the first tie wins) updates `run_idx = i + block_idx, run_max = block_max`.
- Under tensor parallelism, each rank packs its local argmax index into one extra FP32 column in the existing statistics `all_gather`, and the first rank holding the global maximum (or `NaN`) supplies the global index while keeping loss and gradients bit-identical.
- In `group_quantize_mxfp8.cuh` (`76ae4b0981`), setting `tensor_base_for_scales = is_single_tensor ? tensor_start_offset : 0` eliminates the double scale offset.

### 2.2 LMCache 2-D Batch Copy (`cudaMemcpy3DBatchAsync`) for Block-Outermost Layouts (`0d441be069`)

In `csrc/cuda/mp_mem_kernels.cu`, `uniform_layer_pitch(addr, spec)` checks whether consecutive layers sit at a constant byte distance $\text{pitch}$ that fits within a single block stride:
$$\text{pitch} \ge \text{block\_bytes} \ \land\ (N_L - 1) \cdot \text{pitch} + \text{block\_bytes} \le \text{block\_stride}$$
When satisfied (as in vLLM's `BLNHC`), `execute_direct_copy_transfer` issues a single 2-D copy (`width = block_bytes, height = nl`) per `(kv_plane, block)` via `cudaMemcpy3DBatchAsync`, reducing copy descriptors by $N_L\times$; for layer-outermost layouts, it returns `0` and falls back to 1-D copies.

### 2.3 NIXL Multipart S3-over-RDMA & GPUNetIO Two-SGE WRITE (`99d469be5d` & `14e329032b`)

In `src/utils/object/rdma/rdmaRetry.cpp`:
- **PUT (`rdmaPutObject`)**: Objects larger than $4\text{ GiB} - 64\text{ KiB}$ initiate an S3 multipart upload (`cp.createMultipartUpload`) and split the payload into $\lceil S / (4\text{ GiB} - 64\text{ KiB}) \rceil$ RDMA `UploadPart` requests, each acquiring its own token from the shared registration and aborting the multipart upload on any failure.
- **GET**: Large reads issue consecutive ranged RDMA GETs of up to $4\text{ GiB} - 64\text{ KiB}$ each.
- **GPUNetIO (`14e329032b`)**: Supports peer-owned VRAM (`cudaDeviceGetAttribute` / peer context) and pairs eligible WRITE descriptors into two-SGE WQEs without a staging gather buffer.

---

## 3. Deterministic Verification (`experiments/te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma`)

Running `python experiments/te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma/run.py --check` validates all 8 invariants:
- **TransformerEngine**: Saves `131,334,144 B` (`125.25 MiB`) of HBM reads per rank at $T=4096, V=128,256, \text{TP}=8$ against `16,384 B` of extra all-gather payload (**8016.0×** traffic reduction ratio), verifies NaN-first (`idx=2`, `idx=4`) and left tie-break (`idx=1`) semantics, and fixes the MXFP8 scale offset (`2048` vs buggy `4096`).
- **LMCache 2-D Copy**: Collapses 61-layer 32-block `BLNHC` transfers from `1952` 1-D descriptors to `32` 2-D descriptors (**61.0× reduction**).
- **NIXL S3-over-RDMA & GPUNetIO**: Splits `9 GiB` into `3` parts and `64 GiB` into `17` parts (all $\le 4,294,901,760\text{ B}$), and halves GPUNetIO WQEs from `64` to `32` via two-SGE WRITE pairing.
