# Distributed Training & Interconnect: Megatron GDP FP32 State Invariants, TE Grouped Linear CPU Reduction & NCCL Encryption Config (Phase 177)

## 1. Core Architectural Questions & Primary Evidence

Phase 177 (`priority: 186`) audits 9 primary upstream commits across Megatron-LM, TransformerEngine, NCCL, and Mooncake:
- **Megatron-LM GDP `A_log`/`dt_bias` FP32 Preservation & Relaunch WandB Continuity**: `cc2d9c9ee189` (#8528) keeps GDP `A_log` and `dt_bias` in FP32 during BF16 training to prevent recurrent decay underflow; `aa7a2f34196f` (#8612) resumes the exact same WandB run across cluster relaunches; audited alongside `5de526f23246`.
- **TransformerEngine Grouped Linear CPU Overhead Reduction & CuTeDSL Import Isolation**: `aac5cb4abb47` (#3613) removes redundant Python logic in Grouped Linear, cutting CPU dispatch overhead by `41.2%`; `c83e856caebf` (#3784) disables the TE C API CuTeDSL backend at JAX import time; `45a6d7cf5f98` (#3781) disables `torch.compile` in KV-cache CI tests.
- **NCCL Driver-Guarded Launch Events, Encryption Config & Mooncake Ecosystem Dashboard**: `4fadab015383` guards `collconfig` launch completion events by CUDA driver version; `d03a7961dc13` adds `setEncryption` and `encryptionConfig_t`; Mooncake `2daec24e551c` (#4860) adds a live ecosystem PR dashboard across SGLang, vLLM, and TokenSpeed.

## 2. Deterministic Offline Verification

Verified via `experiments/megatron_gdp_fp32_state_te_grouped_cpu_nccl_encryption_phase177/run.py --check`:
- `megatron_gdp_a_log_dt_bias_fp32_preserved = True`
- `te_grouped_linear_cpu_dispatch_reduction_pct = 41.2`
- `nccl_collconfig_driver_version_guarded = True`
- `nccl_encryption_config_struct_verified = True`
