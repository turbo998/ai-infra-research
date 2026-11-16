# Round 305: ZeRO-Offload / FSDP CPUOffload Async PCIe Overlap & Layerwise SIMD `AVX-512 CPUAdam` Selection

Deterministic evaluator comparing 4 CPU optimizer/gradient offload architectures (`48 GB/GPU` offloaded state on `8x H100`) across step latency (`s`), GPU HBM footprint (`GB`), and effective `MFU` (`%`).
