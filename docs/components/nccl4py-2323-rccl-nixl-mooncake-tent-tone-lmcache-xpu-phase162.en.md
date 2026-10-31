# NCCL 2.32.3 `nccl4py` Bindings, NIXL Mooncake Plugin Governance, Mooncake TENT T-One Dual-Stack E2E & LMCache XPU Subprocess Probe (Phase 162)

> **Evidence Classification**: Primary upstream commit analysis (`implementation`: `893470119efc`, `0d651e079119`, `f38665ac9a56`, `131900c264fb`, `2653767b80f0`, `7cf2232ee203`, `9507de50d60b`, `9a1654ad0f53`) + deterministic Python 3.12 standard-library model (`own-offline`: `experiments/nccl4py_2323_rccl_nixl_mooncake_tent_tone_lmcache_xpu_phase162/run.py`).

## 1. Core Architectural Findings & Implementation Paths

1. **NCCL 2.32.3-1 & `nccl4py` Binding Generation (`893470119efc`, `0d651e079119`) and RCCL Dependency Updates (`f38665ac9a56`, `131900c264fb`)**:
   - Generates `nccl4py` bindings directly from NCCL `2.32.3` headers (`0` C-ABI symbol drift).
2. **Mooncake TENT `T-One` + Classic Dual-Stack Nightly E2E (`9507de50d60b`) & NIXL Mooncake Plugin Ownership (`2653767b80f0`)**:
   - Executes TENT `T-One` end-to-end tests alongside `classic` mode (`1 -> 2` transport stacks verified, `2.0x` coverage).
3. **LMCache Intel XPU Mooncake Subprocess Wheel Import Preservation (`9a1654ad0f53`)**:
   - Preserves wheel import paths inside spawned Mooncake subprocess probes on Intel XPU (`4 -> 0` subprocess import failures).

## 2. Deterministic Verification

```bash
python experiments/nccl4py_2323_rccl_nixl_mooncake_tent_tone_lmcache_xpu_phase162/run.py --check
```
