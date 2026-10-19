# CUTLASS SM100 Ptr-Array / Grouped GEMM (`TensorMapUpdater` + `CLC`) 算例

复现 `NVIDIA/cutlass` (`sm100_gemm_array_tma_warpspecialized.hpp` 与 `sm100_tile_scheduler_group.hpp`) 的异步 `TensorMapUpdater` 双级 `clc_response` 描述符流水线与 Blackwell `Cluster Launch Control (CLC)` 动态工作窃取收益。

## 运行

```powershell
python experiments\cutlass_sm100_grouped_clc_gemm\run.py --check
```
