# NCCL 2.32.3, RCCL DMABUF Disable & amd-smi Migration, NIXL GPUNetIO, Mooncake Loopback & LMCache gRPC (Phase 161)

Deterministic standard-library analytical model (`own-offline`) for Phase 161 (`priority: 170`) Track 3:
- **NCCL / RCCL**: `9f99e8b8c2b5`, `12df1a11afad`, `9a443f305499` (respect user DMABUF disable), `cd745b1f4b94` (`amd-smi` migration from `rocm-smi`).
- **NIXL / Mooncake / LMCache**: `ca2946d0fd1c` (GPUNetIO UCX submodule auth), `f3d81669b41b` (NPU `-j2` build cap), `e174114d221c` (HiCache loopback metadata bind), `acaca416ee3c` (real gRPC server e2e test).
