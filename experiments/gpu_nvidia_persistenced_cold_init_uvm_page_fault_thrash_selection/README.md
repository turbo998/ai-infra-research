# Round 315: GPU `nvidia-persistenced` Cold-Init Penalty & `UVM` (`cudaMallocManaged`) Page-Fault Thrashing Selection

Deterministic evaluator comparing 4 GPU driver persistence and memory oversubscription architectures on `8x H100` across `cudaInit()` latency (`s`), effective PCIe bandwidth (`GB/s`), and step time (`ms`).
