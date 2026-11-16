# PyTorch Host-to-Device (`H2D`) `.to("cuda", non_blocking=True)` Pageable Fallback vs. Pinned Ring-Buffer & `h2d_stream` Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-h2d-non-blocking-pageable-fallback-pinned-ring-buffer-selection-faq.md`:
- **Tier 1 (Pageable CPU Tensor `.to("cuda", non_blocking=True)`)**: Silently falls back to synchronous CPU staging copy inside `cuMemcpyHtoDAsync`, stalling the Python thread for `8.40 ms/step` (`36.5%` GPU idle bubble).
- **Tier 2 (Dynamic Per-Step `tensor.pin_memory().to("cuda", non_blocking=True)`)**: Triggers OS `mmap_lock` / `cudaHostAlloc` page-locking syscalls every step, stalling for `24.60 ms/step` (`62.4%` GPU idle bubble).
- **Tier 3 (Single Persistent Pinned Buffer without `cudaEvent` Reuse Guard)**: `2.35 ms` exposed stall on the default stream and `14.2%` silent data corruption when CPU overwrites the pinned buffer while DMA is still reading it.
- **Tier 4 (Pre-Allocated Pinned Double-Buffered Ring `N=2` + Coalesced Single-Slab Packing + Dedicated `h2d_stream` & `cudaEvent` Guard)**: `0.06 ms` exposed stall (`140.0x` faster than Tier 1, `410.0x` faster than Tier 2), `0.4%` GPU bubble, `0.0%` data corruption.
