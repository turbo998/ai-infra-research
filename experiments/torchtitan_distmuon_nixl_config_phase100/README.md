# TorchTitan DistMuon Accelerator Staging, NIXL Config Isolation & Mooncake Msgpack-C 6 Timestamps (Phase 100)

Deterministic Python 3.12 standard-library model reproducing:
1. **TorchTitan `#5352` (`cb0f6f3328`)**: `DistMuon(ns_device="accelerator" | "cuda" | None)` staging 2D compute views from host DRAM to the GPU for 5-step Newton-Schulz orthogonalization under FSDP2 CPU offload while keeping parameters, momentum, and Gloo redistribution on host (`~15.0s -> ~1.0s` per step, `atol <= 1e-5`).
2. **NIXL `#2390` (`c072fb28d0`) & `#2307` (`0fe526d1aa`)**: `nixl_agent_config(backends=None)` per-instance `["UCX"]` list isolation (preventing cross-agent mutation/clearing) and `nixlbench` `--filepath` directory validation prior to GDS/POSIX multi-process launch.
3. **Mooncake `#4906` (`12a7e9f10a`) & LMCache `#5779` (`d44e364fb3`)**: Fractional float64 timestamp preservation under `msgpack-c 6` and unified shared bearer/API-key authentication.
