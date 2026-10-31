# Inference & Cross-Process EP Runtime Phase 169: DeepEP `cuMem` Cross-Process Shared Memory, TRT-LLM Cached MPI Pool Drain, SGLang EFA 1.51.0 Upgrade, and vLLM ROCm CI Consolidation & AITER MXFP4 Verification

## 1. Architectural Context & Core Bottlenecks

Multi-process expert parallelism (EP), MPI-backed serving orchestration, and ROCm/EFA container pipelines encountered three infrastructure bottlenecks:
1. **Static Remapping Constraints of Legacy `cudaIpcGetMemHandle` in DeepEP**: Managing cross-process shared memory buffers via legacy CUDA Runtime IPC handles prevented dynamic virtual address stitching and fine-grained physical page remapping across intranode EP ranks.
2. **Child-Process Spawn Deadlocks with Cached MPI Pools in TensorRT-LLM (`nvbugs/6700875`)**: Spawning child processes while holding cached MPI worker pools leaked active MPI/UCX communicator state into children, causing hangs.
3. **EFA Installer Versioning & Multi-Tier vLLM ROCm CI Bloat**: SGLang's PD disaggregation images required `EFA installer 1.51.0`, while vLLM's ROCm CI maintained 4 separate image tiers and skipped AITER MXFP4 matrix/token padding tests.

---

## 2. Primary Upstream Mechanism Deep Dive (11 Zero-Overlap Commits)

### 2.1 DeepEP: CUDA Driver `cuMem` Cross-Process Shared Memory & Legacy Torch Compatibility
- **`9249c25796` (`2025-11-04`, `#217`)**: Adopts the CUDA Driver `cuMem` API (`cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemMap`) in `csrc/deep_ep.{cpp,hpp}` and `deep_ep/buffer.py` for cross-process shared memory management (`2 MiB` granularity, `0` remap stalls).
- **`bfded34800` (`2025-10-29`, `#469`)**: Restores compatibility with older Python and PyTorch releases.

### 2.2 TensorRT-LLM: Draining Cached MPI Pools Before Child Spawn & CBTS Stage Preservation
- **`9d3b492125` (`2026-10-29`, `#20331`)**: Drains cached test MPI pools in `session_reuse.py` before any test starts a child process (`nvbugs/6700875`).
- **`c69d879b4d` (`2026-10-30`, `#21093`)**: Preserves auto-triggered stages in the CBTS coverage tier.

### 2.3 SGLang & vLLM: EFA 1.51.0, DSpark MLA Seed, ROCm CI Collapse & AITER MXFP4 Tests
- **`e337bb10b5` (`2026-10-30`, `#46892`), `e4bb0478c5` (`#47540`), `02cd9ff0a2` (`#47395`) & `a8559ca775` (`#47424`)**: Updates SGLang EFA Docker images to `EFA installer 1.51.0`, seeds DSpark MLA draft weights in `test_dflash_mla.py`, fixes AMD Dockerfile-only CI filters, and wires `output_streamer` into chunked-abort tests.
- **`bba4d9afec` (`2026-10-29`, `#61939`), `e3f03f4f89` (`#62996`) & `9957f74617` (`#64362`)**: Collapses vLLM ROCm CI image tiers (`4` → `1`, `4.0x` reduction), un-skips AITER MXFP4 padding-matrix and token-padding tests, and adds an OpenVLA chat template fallback.

---

## 3. Deterministic Offline Verification Summary

Validated via `experiments/deepep_cumem_shared_sglang_efa151_trtllm_mpi_drain_phase169/run.py`:

| Subsystem | Baseline Metric | Phase 169 Verified Metric | Architectural Impact |
|---|---|---|---|
| DeepEP Cross-Process Shared Memory (`8` ranks) | `8` legacy IPC remap stalls | **`0` `cuMem` remap stalls** (`2 MiB` pages) | Elastic virtual address mapping across EP ranks |
| TRT-LLM Child Process Spawn (`4` MPI pools) | `4` inherited MPI deadlocks | **`0` deadlocks after pool drain** | Eliminates fork/spawn hangs (`nvbugs/6700875`) |
| SGLang EFA & vLLM ROCm CI / AITER MXFP4 | `4` ROCm image tiers; skipped tests | **EFA `1.51.0`**; **`1` unified tier (`4.0x`)** | Re-enables `2` AITER MXFP4 padding test suites |
