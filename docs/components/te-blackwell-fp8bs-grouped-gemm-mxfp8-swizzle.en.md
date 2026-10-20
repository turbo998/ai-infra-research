# TransformerEngine Blackwell FP8 Block-Scaling Grouped GEMM MXFP8 Swizzle Emulation (`grouped_swizzle_block_scaling_to_mxfp8_kernel`) and NVFP4 Transpose Quantization

> **中文版**: [te-blackwell-fp8bs-grouped-gemm-mxfp8-swizzle.md](te-blackwell-fp8bs-grouped-gemm-mxfp8-swizzle.md) | **Experiment**: [`experiments/te_grouped_mxfp8_swizzle_sim/README.md`](../../experiments/te_grouped_mxfp8_swizzle_sim/README.md)

### Executive Summary

Because `cublasLt` on NVIDIA Blackwell (SM100+) does not natively support Hopper's FP32 1D (`1x128`) and 2D (`128x128`) FP8 block-scaling GEMM recipes, TransformerEngine (commit `f4a69dc81e`) emulates FP8 block scaling via MXFP8 grouped GEMM by **sharing the rowwise FP8 data pointer zero-copy (`output->data.dptr == input->data.dptr`)** and converting compact FP32 `scale_inv` buffers into `128x4 = 512 B` interleaved MXFP8 `E8M0` scale tiles (`kTileOutputBytes = 512`) inside `grouped_swizzle_block_scaling_to_mxfp8_kernel`. Using **device-side 32-lane `__shfl_up_sync` exclusive prefix-sum tables (`tile_start, in_start, out_start`) in dynamic shared memory paired with binary search (`find_tensor`)**, the kernel remains 100% CUDA-Graph safe when expert token counts vary across replays, bounding host-allocated scale buffers for 256-expert DeepSeek-V3 MoE FPROP to `14,622,720 B` (`1.2944x` slack over exact `11,296,768 B`) and opting into `>48 KiB` dynamic shared memory via `cudaFuncSetAttribute` when `num_tensors >= 2048` (supporting up to `9,685` grouped tensors on Blackwell).

### Upstream Source Responsibility & Locations

| Module & File | Upstream Repo & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `transformer_engine/common/include/transformer_engine/swizzle.h` | `NVIDIA/TransformerEngine` commit `f4a69dc81eabb5002a662e56a36e8454c9840a92` / blob `f85d88fdce83eb09253449fa3f98fa01e5522f51` | Declares C ABI entrypoints `nvte_swizzle_block_scaling_to_mxfp8_scaling_factors`, multi-tensor batching, and CUDA-Graph-safe `nvte_swizzle_grouped_block_scaling_to_mxfp8_scaling_factors`. |
| `transformer_engine/common/swizzle/swizzle_block_scaling.cu` | `NVIDIA/TransformerEngine` commit `f4a69dc81eabb5002a662e56a36e8454c9840a92` / blob `6cfa676592b02f21fefd9f60135ac5c9bd1b0e4f` | Implements `grouped_swizzle_block_scaling_to_mxfp8_kernel`, binary search `find_tensor`, dynamic SMEM opt-in `enable_table_smem`, and host upper bound `grouped_mxfp8_scale_bytes_upper_bound`. |
| `transformer_engine/pytorch/csrc/extensions/swizzle.cpp` | `NVIDIA/TransformerEngine` commit `f4a69dc81eabb5002a662e56a36e8454c9840a92` / blob `03eec7d80fd9c48b5cfa0758f17f470b62f7c422` | Implements PyTorch C++ extensions `swizzle_scales_for_gemm` and grouped block-scaling conversion wrappers with 16-byte pointer alignment and `with_gemm_swizzled_scales` tracking. |
| `transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh` | `NVIDIA/TransformerEngine` commit `4a3a8a8e8578e96319467d0cc96b9f5db552a7d3` / blob `f83564b35fa5cb178f8bbd6333625d6e83842e73` | Implements SM100 NVFP4 `compute_rowwise_amax_kernel` / `compute_columnwise_amax_kernel` (`ptx::abs_max_2x` vectorized reduction) and `wmma.load/store.m16n16k16.shared::cta` shared-memory transpose. |

### Core Implementation Path, Memory Layout & Synchronization Contracts

1. **Why Blackwell Emulates FP8 Block-Scaling via MXFP8 Interleaved Scales (`swizzle.h:87-145`, `swizzle_block_scaling.cu:524-589`)**:
   - On Hopper (SM90), `cublasLt` natively supports `1x128` (1D) and `128x128` (2D) FP32 `scale_inv` factors. On Blackwell (SM100), 5th-generation Tensor Cores (`tcgen05.mma`) natively execute **MXFP8 1D Scaling** (one 8-bit `E8M0` power-of-two scale per 32 elements stored in `128x4` interleaved tiles).
   - Because `128 = 4 * 32`, every `128x128` FP8 data tile maps to a `128 x 4 = 512 B` tile of `E8M0` scales (`kTileOutputBytes = 512`). By aliasing the rowwise FP8 data buffer directly (`output->data.dptr == input->data.dptr`) and replicating/converting each FP32 block scale into the 4 corresponding `E8M0` scale columns in `128x4` swizzled order, Blackwell executes Hopper FP8 block-scaling models at native MXFP8 Tensor Core speed.
2. **CUDA-Graph-Safe Device Prefix Sums and Binary-Search Tile Mapping (`swizzle_block_scaling.cu:312-442`)**:
   - In MoE grouped GEMM, per-expert dimensions (`first_dims` or `last_dims`) reside in device memory and change on every CUDA Graph replay.
   - Inside `grouped_swizzle_block_scaling_to_mxfp8_kernel`, Warp 0 (`threadIdx.x < 32`) steps through `num_tensors` in chunks of 32, using 5-stage `__shfl_up_sync(0xFFFFFFFF, sum, d)` butterfly scans to populate three exclusive prefix-sum tables in dynamic shared memory: `tile_start[num_tensors + 1]`, `in_start[num_tensors]`, and `out_start[num_tensors]`.
   - After `__syncthreads()`, warps iterate over `tile_id < total_tiles` with stride `gridDim.x * WARPS_PER_TB`, resolving the owning tensor $t$ in $O(\log_2 N)$ via binary search `find_tensor(tile_start, num_tensors, tile_id)`. Zero-token experts have `tile_start[t] == tile_start[t+1]` and are automatically skipped.
3. **Dynamic Shared Memory 48 KiB Opt-In and Host-Side Upper Bound (`swizzle_block_scaling.cu:358-360, 447-489`)**:
   - Shared memory table size is `table_smem_bytes(n) = (3 * n + 1) * 8 = 24n + 8` bytes. Up to `2,047` tensors (`49,136 B`) fit in the default `48 KiB` (`49,152 B`) limit; at `n >= 2048` (`49,160 B`), `enable_table_smem` invokes `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_bytes)` once per device, scaling up to `9,685` tensors on Blackwell (`227 KiB`).

### Key Equations, Units & Engineering Assumptions

1. **Per-Tensor Input FP32 Scale Floats & Output MXFP8 E8M0 Bytes**:
   $$\text{in\_floats}_{\text{1D}}(r, c) = \left\lceil \frac{c}{128} \right\rceil \cdot \operatorname{roundup}(r, 4), \qquad \text{in\_floats}_{\text{2D}}(r, c) = \left\lceil \frac{r}{128} \right\rceil \cdot \operatorname{roundup}\!\left(\left\lceil \frac{c}{128} \right\rceil, 4\right)$$
   $$\text{out\_bytes}_{\text{MXFP8}}(r, c) = \operatorname{roundup}(r, 128) \cdot \left\lceil \frac{c}{128} \right\rceil \cdot 4 = \text{tiles}(r, c) \times 512\text{ B}$$
2. **Host-Side Synchronization-Free Upper Bound (`swizzle_block_scaling.cu:466-489`)**:
   - **Uniform**: $B_{\text{uniform}} = n \cdot \operatorname{roundup}(F / n, 128) \cdot \lceil L / 128 \rceil \cdot 4$ (exact)
   - **Varying First Dim ($\sum r_t = F$)**: $B_{\text{varying\_first}}(F, L, n) = (F + 127n) \cdot \lceil L / 128 \rceil \cdot 4$
   - **Varying Last Dim ($\sum c_t = L$)**: $B_{\text{varying\_last}}(F, L, n) = \operatorname{roundup}(F, 128) \cdot 4 \cdot (\lceil L / 128 \rceil + n)$

### Deterministic Verification (`experiments/te_grouped_mxfp8_swizzle_sim`)

Executed via `python experiments/te_grouped_mxfp8_swizzle_sim/run.py --check` (`experiments/te_grouped_mxfp8_swizzle_sim/results/summary.json`):

| Scenario | Tensors $n$ | Logical Shape $[F, L]$ | `128x128` Tiles | Dynamic SMEM (`table_smem_bytes`) | Input FP32 Scale Bytes | Exact MXFP8 Scale Bytes | Host Upper Bound (`host_upper_bound_bytes`) | Slack Ratio | Out/In Scale Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ds_v3_256e_1d_fprop` (16 empty experts) | `256` | `[32768, 7168]` | `22,064` | `6,152 B` (`optin=false`) | `7,420,672 B` | `11,296,768 B` | `14,622,720 B` | `1.2944x` | `1.5223x` |
| `ds_v3_256e_2d_fprop` (2D 128x128 blocks) | `256` | `[32768, 7168]` | `22,064` | `6,152 B` (`optin=false`) | `88,256 B` | `11,296,768 B` | `14,622,720 B` | `1.2944x` | **`128.0x`** |
| `ultra_moe_2048e_optin_smem` | `2048` | `[65536, 4096]` | `65,536` | **`49,160 B` (`optin=true`)** | `8,388,608 B` | `33,554,432 B` | `41,680,896 B` | `1.2422x` | `4.0x` |
| `ds_v3_64e_wgrad_varying_k` | `64` | `[7168, 32768]` | `16,072` | `1,544 B` (`optin=false`) | `8,228,864 B` | `8,228,864 B` | `9,175,040 B` | `1.1150x` | `1.0x` |

### Architect Design Exercises & Falsifiable Questions

1. **Why does emulating 2D FP8 block scaling (`128x128` blocks) on Blackwell expand scale buffer size by `128.0x` (`88,256 B` -> `11,296,768 B`), and why is it still net-optimal?**
   - **Answer**: 2D block scaling stores a single 4-byte `float` per `128x128` tile, whereas MXFP8 requires one 1-byte `E8M0` scale per `1x32` block (`128 rows * 4 cols = 512 B` per `128x128` tile, a `512 / 4 = 128x` expansion). Because the 224 MiB FP8 data tensor itself is aliased zero-copy, the 11.3 MB swizzled scale buffer adds only `4.8%` footprint overhead while unlocking SM100 fifth-gen Tensor Core throughput.
2. **Falsifiable Hypothesis**: Can `grouped_mxfp8_scale_bytes_upper_bound` accept a grouped tensor where both `first_dims` and `last_dims` vary dynamically on the device?
   - **Verification**: `swizzle_block_scaling.cu:473-475` explicitly rejects this via `NVTE_CHECK(!(t.first_dims.has_data() && t.last_dims.has_data()))`.
