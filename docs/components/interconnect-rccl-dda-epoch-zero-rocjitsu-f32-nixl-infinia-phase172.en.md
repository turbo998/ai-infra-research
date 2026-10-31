# Interconnect DDA Epoch Zeroing Order, Zero-Selector Stores & DDN Infinia 2.5 Deep Dive (Phase 172): ROCm RCCL DDA Scratch/Epoch Zeroing, `rocjitsu` FLOAT32 Stores & NIXL Infinia 2.5

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
|---|---|---|---|
| `ROCm/rocm-systems` | `e7cb4b477a` | 2026-10-30 | `fix(rccl): order DDA fabric scratch/epoch zeroing before first LL kernel (#14254)` |
| `ROCm/rocm-systems` | `ce3fa320bd` | 2026-10-30 | `[rocjitsu] Fix zero-selector FLOAT32 buffer stores and enable guitar-hero coverage (#13988)` |
| `NVIDIA/nccl` | `59a784c3d9` | 2026-09-10 | `nccl4py: generate bindings from in-tree NCCL headers` |
| `NVIDIA/nccl` | `76c747405b` | 2026-09-10 | `Improve nccl4py packaging and versioning` |
| `ai-dynamo/nixl` | `5bca2560f8` | 2026-10-06 | `DDN Infinia 2.5 Updates (#2357)` |
| `ai-dynamo/nixl` | `7c8d82c1b2` | 2026-10-07 | `EXAMPLES/DEVICE/EP: Removed high-throughput kernels. (#2345)` |
| `kvcache-ai/Mooncake` | `46061cb23a` | 2026-10-16 | `[PR] Add priority declaration (#4718)` |
| `LMCache/LMCache` | `8bfcdfdea4` | 2026-10-22 | `fix: make foreign tools import independent of safe path (#5986)` |

---

## 2. Architectural Analysis

### 2.1 ROCm RCCL DDA Fabric Scratch/Epoch Zeroing (`e7cb4b477a`) & `rocjitsu` FLOAT32 Buffer Stores (`ce3fa320bd`)
- **DDA Scratch/Epoch Zeroing Before First LL Kernel (`#14254`)**: Orders Direct Data Access (DDA) fabric scratch/epoch zeroing strictly before the first Low-Latency (LL) collective kernel launch, eliminating uninitialized epoch flag races (`0` races).
- **`rocjitsu` Zero-Selector FLOAT32 Store Fix (`#13988`)**: Fixes FLOAT32 buffer stores with zero selectors in `rocjitsu` (`0` lane corruptions).

### 2.2 `nccl4py` In-Tree Header Bindings (`59a784c3d9`) & NIXL DDN Infinia 2.5 (`5bca2560f8`)
- **`nccl4py` In-Tree ABI Synchronization**: Generates `nccl4py` bindings directly from in-tree NCCL headers (`0` ABI mismatches), updates NIXL to DDN Infinia `2.5` (`#2357`), and decouples LMCache foreign tools imports from safe-path restrictions (`#5986`).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/rccl_dda_epoch_zero_rocjitsu_f32_nixl_infinia25_phase172/run.py --check`:
- `uninitialized_dda_epoch_ll_races`: **`0`** (`rocjitsu_zero_sel_f32_store_corruptions = 0`)
- `nccl4py_header_abi_mismatches`: **`0`** (`ddn_infinia_sdk_version = "2.5"`)
- `lmcache_foreign_import_path_failures`: **`0`**
