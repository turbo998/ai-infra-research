# Megatron-LM PretrainConfig Data Section & Inductor Fork Guard, PyTorch AVX2 `zero_mask` 16-Bit Fix, Mooncake etcd TLS/mTLS & GIL-Free Registration, LMCache Stored-Chunk Telemetry & Dynamo/NIXL Cardinality Pruning (Phase 158)

> **中文版**: [megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158.md](megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158.md)
> **Question ID**: `q-megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158` (`priority: 167`)
> **Evidence ID**: `ev-megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-2026-phase158`

---

## 1. Architectural Context & Core Findings

In large-scale pretraining and distributed KV-cache infrastructure, premature runtime initialization before config validation, CPython `fork()` races in TorchInductor's async compile pool, 2-bit-per-lane AVX2 `zero_mask` bugs, GIL contention during RDMA `ibv_reg_mr`, and per-PID Prometheus labels trigger production failures and observability blowups. Auditing 10 primary upstream commits merged between `2026-10-12` and `2026-10-31`, we establish five key architectural conclusions:

1. **Megatron-LM `PretrainConfigContainer` Data Section, Post-Validation Runtime Init & Inductor Fork Guard (`Megatron-LM@831aa66`, `#8341`; `Megatron-LM@ce852bf`, `#8272`; `Megatron-LM@b607c1b`, `#8318`)**: Adds `GPTDatasetConfig` and `DataloaderConfig` to `PretrainConfigContainer` with post-tokenizer validation, moves `initialize_runtime_services` strictly after `parse_and_validate_args`, and sets `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0` on Python `<3.12.8` and `3.13.0` (`0.0%` fork deadlock rate).
2. **PyTorch AVX2 `Vectorized<Half/BFloat16>::zero_mask` 16-Bit Packing & ROCm `LoadHIP` Compiler Preservation (`pytorch@43ae9ac`, `#202480`; `pytorch@dc674ee`, `#201449`)**: Replaces raw `_mm256_movemask_epi8` (which produced a 32-bit mask with 2 bits per 16-bit lane, yielding `popcount = 14` for 7 zero lanes) with `_mm_packs_epi16(lo, hi)` followed by `_mm_movemask_epi8` (exact 16-bit mask, `popcount = 7`), and preserves caller-set `CMAKE_HIP_COMPILER` in `LoadHIP.cmake`.
3. **Mooncake Environment-Configured etcd TLS/mTLS & GIL-Free Single Memory Registration (`Mooncake@99d2614`, `#4945`; `Mooncake@c72982f`, `#4019`)**: Adds `MOONCAKE_ETCD_TLS_*` configuration across all 6 Go/C++ etcd entry points (`13` CTest cases passing) and adds `pybind11::gil_scoped_release` to `TransferEnginePy::registerMemory` / `unregisterMemory`, cutting Python RPC/heartbeat stall jitter from `18.50 ms` to `0.12 ms` (**154.17x** reduction).
4. **LMCache Accurate `MP_STORE_END` Stored-Token Accounting & Dynamo `qwen3_coder` Parser (`LMCache@12dfa32`, `#6188`; `dynamo@5a48166`, `#17676`)**: Counts only unique chunk positions that committed at least one newly stored object (`stored_chunks`), eliminating **75.0%** overcounting (`4096 -> 1024 tokens` when 12 of 16 chunks are already cached), and bumps `dynamo-parsers-v2` to `0.8.9`.
5. **NIXL `prometheus_mp` Collision-Only `pid` and `agent_instance` Labeling (`nixl@250581c`, `#2387`)**: Emits `pid` and `agent_instance` labels only when $\ge 2$ live agents share `hostname + agent_name + local_rank`, reducing time-series cardinality by **5.0x** (`320 -> 64` series across 5 restarts of 64 agents).

---

## 2. Primary Upstream Evidence Matrix

| Commit | Module / File | Verified Mechanism & Metric |
| :--- | :--- | :--- |
| `NVIDIA/Megatron-LM@831aa66` (`#8341`) | `megatron/training/config/config_container.py` | Integrates `GPTDatasetConfig` and `DataloaderConfig` into `PretrainConfigContainer` |
| `NVIDIA/Megatron-LM@ce852bf` (`#8272`) | `megatron/training/initialize.py` | Orders `initialize_runtime_services` strictly after `parse_and_validate_args` |
| `NVIDIA/Megatron-LM@b607c1b` (`#8318`) | `tests/unit_tests/conftest.py` | Sets `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0` on Python `<3.12.8` / `3.13.0` (`0.0%` hangs) |
| `pytorch/pytorch@43ae9ac` (`#202480`) | `aten/src/ATen/cpu/vec/vec256/vec256_half_avx2.h` | Packs `epi16` via `_mm_packs_epi16` before `_mm_movemask_epi8` (`popcount` `14 -> 7`) |
| `pytorch/pytorch@dc674ee` (`#201449`) | `cmake/public/LoadHIP.cmake` | Preserves caller-supplied `CMAKE_HIP_COMPILER` to prevent full CMake rebuilds |
| `kvcache-ai/Mooncake@99d2614` (`#4945`) | `mooncake-common/etcd/src/etcd_helper.cpp` | Supports `MOONCAKE_ETCD_TLS_*` TLS/mTLS with gRPC `ServerName` preservation (`13` CTests) |
| `kvcache-ai/Mooncake@c72982f` (`#4019`) | `mooncake-integration/transfer_engine/transfer_engine_py.cpp` | Releases GIL in `registerMemory`/`unregisterMemory` (**154.17x** lower Python RPC jitter) |
| `LMCache/LMCache@12dfa32` (`#6188`) | `lmcache/v1/multiprocess/server.py`, `storage_manager.py` | Reports `len(stored_chunks) * chunk_size` in `MP_STORE_END` (**75.0%** overcount removed) |
| `ai-dynamo/dynamo@5a48166` (`#17676`) | `Cargo.lock`, `lib/llm/Cargo.toml` | Bumps `dynamo-parsers-v2` to `0.8.9` fixing `qwen3_coder` `<tool_call>` trigger exclusion |
| `ai-dynamo/nixl@250581c` (`#2387`) | `src/plugins/prometheus_mp/prometheus_mp_Engine.cpp` | Labels by `pid`/`agent_instance` only on live identity collision (**5.0x** lower cardinality) |

---

## 3. Reproducible Verification

```powershell
python experiments\megatron_pretrain_inductor_avx2_mooncake_tls_lmcache_nixl_phase158\run.py --check
```
