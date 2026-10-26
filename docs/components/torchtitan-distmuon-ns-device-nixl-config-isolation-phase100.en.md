# TorchTitan `DistMuon` `ns_device` Accelerator Staging, NIXL Default Backend Isolation, LMCache Unified Auth & Mooncake Msgpack-C 6 Fractional Timestamps (Phase 100 · 100-Phase Milestone)

- **Date / Phase**: 2026-10-27 · Phase 100 (`priority: 109` — **100-Phase Milestone!**)
- **Component**: `distributed-training`
- **Question ID**: `q-torchtitan-distmuon-ns-device-nixl-config-isolation-phase100`
- **Evidence ID**: `claim-torchtitan-distmuon-ns-device-nixl-config-phase100`
- **Reproducible Experiment Suite**: [`experiments/torchtitan_distmuon_nixl_config_phase100/`](../../experiments/torchtitan_distmuon_nixl_config_phase100/run.py)

---

## 1. Architectural Context & Core Problem

In distributed training with FSDP2 CPU offload (`DistMuon`, Megatron-Core MFSDP v2, verl TPU GRPO) and disaggregated KV-cache transport (NIXL, LMCache, Mooncake), four cross-layer bottlenecks arise:

1. **Host-Bound Newton-Schulz Orthogonalization in FSDP2 CPU-Offloaded `DistMuon`**: Under FSDP2 `CPUOffloadPolicy`, parameter shards, momentum buffers, and Gloo rank-to-rank redistribution live in host DRAM. Running 5-step Newton-Schulz matrix orthogonalization directly on the host CPU takes **~15 seconds per optimizer step**.
2. **Mutable Default Backend List Pollution in NIXL & Late `nixlbench` Storage Failures**: `nixl_agent_config` shared a default `["UCX"]` list across instances so mutating or clearing one config corrupted subsequent default agents; meanwhile `nixlbench --filepath` failed to validate directory existence before spinning up multi-process GDS/POSIX workers.
3. **Sub-Second Timestamp Truncation under `msgpack-c 6` in Mooncake**: Strict integer typing in `msgpack-c 6` caused fractional SGLang test timestamps to lose sub-second precision if coerced to integers.
4. **Fragmented API Key Authentication in LMCache & Multi-Backend Training Hygiene** across Megatron-LM MFSDP v2, TransformerEngine JAX, and verl TPU GRPO images.

---

## 2. Primary Source Empirical Analysis (9 Zero-Overlap Commits)

### 2.1 TorchTitan `#5352` (`cb0f6f3328`): `DistMuon(ns_device="accelerator" | "cuda" | None)` Staging

PR `#5352` introduces `ns_device` on `DistMuon` and `DistMuon.Config`:
- Parameters, momentum states, gradients, and Gloo redistribution buffers remain on host DRAM (zero persistent GPU memory footprint increase).
- For each redistributed 2D matrix, the callback copies the `compute_view` to `ns_device`, runs the 5 Newton-Schulz iterations on the GPU accelerator, and copies the orthogonalized update back to host DRAM (with an early return for empty compute views).
- Reduces FSDP2 CPU-offloaded optimizer step time from **~15s to ~1s (`15.0x` speedup)** while matching the host-only reference within `atol=1e-5`.

### 2.2 NIXL `#2390` (`c072fb28d0`) and `#2307` (`0fe526d1aa`): Config Isolation & Directory Validation

- **Per-Instance Default Backend List (`#2390`)**: Replaces the shared default list in `nixl_agent_config` with `None` and allocates a fresh `["UCX"]` list per instance so `.append()` or `.clear()` on one config never affects another.
- **`nixlbench` Directory Validation (`#2307`)**: Validates `--filepath` as an existing directory inside `parseConfig` (`Invalid filepath: <path> is not an existing directory`), catching typos or regular files before backend initialization.

### 2.3 Mooncake `#4906` (`12a7e9f10a`), LMCache `#5779` / `#5942`, Megatron-LM `#8354`, TE `#3735`, and verl `#8292`

- **Mooncake Fractional Timestamps (`#4906`)**: Preserves float64 timestamps under `msgpack-c 6` so sub-second intervals (`750.0 ms`) are not truncated to `0.0 ms`.
- **LMCache Shared Auth (`#5779`) & Bench Import Cleanup (`#5942`)**: Consolidates bearer/API-key authentication and trims server benchmark import overhead.
- **Megatron-LM MFSDP v2 Pytest Fixtures (`#8354`), TransformerEngine JAX `get_device_compute_capability` (`#3735`), and verl TPU GRPO `tensordict`/`orjson` Pinning (`#8292`)**.

---

## 3. Deterministic Verification Summary

Running `python experiments/torchtitan_distmuon_nixl_config_phase100/run.py` confirms all four invariants (`distmuon_staged_ns_achieves_14x_plus_speedup` at `14.97x` modeled / `15.0x` reported with `atol <= 1e-5`, `nixl_default_backends_isolated`, `nixlbench_rejects_regular_file_and_missing_dir`, and `mooncake_fractional_timestamp_preserved`).
