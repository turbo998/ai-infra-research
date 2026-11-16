# PyTorch Autoregressive `torch.cat` Quadratic $O(S^2)$ KV Copy Explosion (`536.87 GB` Traffic) vs. Pre-Allocated `index_copy_` (`StaticCache`) Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-autoregressive-torch-cat-quadratic-kv-copy-preallocated-index-copy-selection-faq.md`:
- **Tier 1 (Per-Step `past_kv = torch.cat([past_kv, new_kv], dim=2)` over `4,096` steps)**: Copies $\frac{S(S+1)}{2} = 8,390,656$ token slices (**`536.87 GB` HBM traffic**), triggers `4,096` dynamic allocations, and takes `168.40 ms`.
- **Tier 2 (Chunked `torch.cat` every 64 steps)**: `8.39 GB` HBM traffic, `64` allocations, `6.80 ms`, incompatible with CUDA Graphs.
- **Tier 3 (Pre-Allocated Static Buffer with Python Integer Slice `kv_buf[:, :, step:step+1] = new_kv`)**: `0.26 GB` traffic, `1.45 ms`, but bakes constant slice offset into CUDA Graph capture.
- **Tier 4 (Pre-Allocated Static Buffer + 1D GPU Tensor `kv_buf.index_copy_(2, cache_position, new_kv)`)**: `0.26 GB` total copy traffic (`2,064.9x` reduction), `0` dynamic allocations, `1.12 ms` (`150.4x` faster than Tier 1), and 100% CUDA Graph safe.
