# Grey Failure & Slow Straggler Detection (`Megatron-LM` + `NCCL Profiler v5`)

Deterministic `own-offline` simulator for **16,384-GPU Synchronous Training Grey Failure Localization and Hot-Spare Swap**, grounded in `NVIDIA/Megatron-LM` (`megatron/core/utils.py` `StragglerDetector`, `README_STRAGGLER.md`) and `NVIDIA/nccl` (`src/include/comm.h` `ncclProgressCountersBlock`, `src/include/plugin/profiler/profiler_v5.h` `ncclProfileKernelCh`).

## Key Findings
- **Single-Node Grey Failure Amplification**: In `16,384x H100` (`TP=8, PP=16, DP=128`, baseline `1,850.0 ms/step`, `54.0%` MFU), a single `900 MHz` thermal-throttled GPU (`1.50x` local GEMM slowdown) drags cluster MFU down by **`32.94%`** (`5,396.9` wasted GPU equivalents); a single RoCE/IB port with optical FEC retry degradation (`3.00x` collective slowdown) drops cluster MFU by **`33.22%`**; compound grey failure drops MFU by **`63.00%`** (`19.98%` MFU).
- **Dual-Sided Telemetry Classification & Hot-Spare Recovery**: Combining local CUDA GEMM delta (`MnEtpt/Rnk`, `MnClk/Rnk`), `get_batch` `DRtt`, and NCCL `ncclProfileKernelCh` `pTimer` transfer skew achieves **`100.0%` root-cause classification accuracy** (`compute` vs `network` vs `pcie` vs `compound`) within a 2-step (`3.70 s`) window and triggers a `4.20 s` NVRx in-process hot-spare swap, lifting 30-day cluster Goodput from **`71.35%`** to **`98.83%`** (`+27.48 pts`).

## Verification
```powershell
python experiments\grey_failure_straggler_detector\run.py --check
```
