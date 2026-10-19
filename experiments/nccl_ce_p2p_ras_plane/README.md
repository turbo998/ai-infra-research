# NCCL P2P Copy Engine (`useMemcpy` / `P2P_CUMEM`) & RAS `ncclGpuProgressCounterMonitor`

Deterministic stdlib reproduction of `NVIDIA/nccl` (`src/transport/p2p.cc`: `useMemcpy`, `p2pSendProxyProgress` `cudaMemcpyDeviceToDevice`, `P2P_CUMEM` multi-segment registration) and `src/ras/` (`progress_monitor.cc` `ncclGpuProgressCounterMonitor` with `cudaStreamCaptureModeRelaxed` `sideStream` and `ras_internal.h` `RAS_COLL_DIAG`).

## Run & Verify

```bash
python experiments/nccl_ce_p2p_ras_plane/run.py --check
```
