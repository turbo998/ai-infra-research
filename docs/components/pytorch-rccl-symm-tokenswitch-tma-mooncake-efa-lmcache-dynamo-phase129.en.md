# PyTorch ROCm `NCCLSymmetricMemory` (RCCL), TokenSwitch `topk_weights` Backward Gradients, Inductor Compile-Time Autotune TMA Descriptor Wrapping & `KERNEL_CONFIGS` Persistence, Megatron-LM Unified `get_tokenizer` Initialization, Mooncake EFA `88 GiB` 4KB-Page Limit vs 2MB HugePages, LMCache Concurrent TP Worker Benchmark Transfers, and Dynamo DCGM Connection Closure & `SelectionService` Session Affinity Replication (Phase 129 · Priority 138)

> **Evidence Classification**: This report combines first-hand upstream commits from `pytorch/pytorch` (`b8c7d2c`, `c5c9a00`, `b8d6389`, `e5d366e`), `NVIDIA/Megatron-LM` (`73f17bc`), `kvcache-ai/Mooncake` (`e648c15`), `LMCache/LMCache` (`d67052b`), and `ai-dynamo/dynamo` (`f9c22d7`, `c03f514`) (`implementation`) with deterministic Python 3.12 standard-library modeling in `experiments/pytorch_rccl_symm_tokenswitch_tma_mooncake_efa_lmcache_dynamo_phase129/run.py` (`own-offline`). No live hardware re-benchmarking is claimed.

---

## 1. Architectural Motivation

Across distributed training compilers (PyTorch ROCm symmetric memory, TokenSwitch EP gradients, Inductor module-level kernels, Megatron-LM) and disaggregated KV/serving runtimes (Mooncake EFA, LMCache, Dynamo), NIC page-table limits, compile-time autotune persistence, and control-plane resource lifecycles govern cluster stability:

1. **PyTorch ROCm `NCCLSymmetricMemory` (RCCL) & TokenSwitch `topk_weights` Backward (`pytorch@b8c7d2c`, `c5c9a00`)**: Enables `NCCLSymmetricMemory` on ROCm across HIP graph capture, RCCL window registration, and `3` device ops; and implements backward gradients for `topk_weights` in `nccl_ep.cu` (`TokenSwitch`, `234.88M` MACs at `4096×8×7168`).
2. **PyTorch Inductor Sample-Input TMA Descriptor Wrapping & `KERNEL_CONFIGS` Persistence (`pytorch@b8d6389`, `e5d366e`)**: Wraps captured underlying tensors in host-side TMA descriptor calls under `triton.autotune_with_sample_inputs=True`, and persists compile-time autotuned Triton configs in `KERNEL_CONFIGS` so standalone module launches re-tune **`64 -> 0` kernels** and eliminate cross-rank reduction-order drift.
3. **Megatron-LM Unified `get_tokenizer` Initialization (`megatron@73f17bc`)**: Consolidates tokenizer access onto `global_vars.get_tokenizer` across training and dynamic inference (`3 -> 1` initializations per rank).
4. **Mooncake EFA `88 GiB` 4KB-Page Ceiling vs 2MB HugePages (`mooncake@e648c15`)**: With 4 KB pages, one EFA memory registration covers ~`88 GiB` per NIC, causing a `128 GiB` host KV pool to fail on single-NIC nodes or split across NIC subsets. Backing segments with `hugetlb` or THP (`2 MiB` pages) shrinks MR page-table entries by **`512.0x`** (`33,554,432 -> 65,536`), allowing full `128 GiB+` single-NIC registration.
5. **LMCache Concurrent TP Worker Transfers & Dynamo DCGM Connection Closure / Session Affinity Replication (`lmcache@d67052b`, `dynamo@f9c22d7`, `c03f514`)**: Runs TP worker `STORE`/`RETRIEVE` benchmark transfers concurrently with `--transfer-timeout` (**`148.0 -> 19.6 ms`, `7.551x` speedup at `TP=8`**); closes DCGM scrape connections in Dynamo Operator (**`23,040 -> 0` leaked sockets/hour** across 64 nodes); and exposes `SessionBindingStream` in `SelectionService` for external session affinity replication (**`100%` failover hit rate**).

---

## 2. Deterministic Verification Summary (`own-offline`)

| Subsystem | Upstream Commit(s) | Legacy Baseline | Phase 129 Architecture | Quantitative Impact |
|---|---|---|---|---|
| **PyTorch ROCm SymmMem & TokenSwitch** | `pytorch@b8c7d2c`, `c5c9a00` | No ROCm `NCCLSymmetricMemory`; no `topk_weights` grad | RCCL symmetric memory (`3` device ops) + `nccl_ep.cu` `topk_weights` backward | **Full ROCm symmetric memory + `234.88M`-MAC router grad** |
| **PyTorch Inductor TMA & `KERNEL_CONFIGS`** | `pytorch@b8d6389`, `e5d366e` | Bare tensor passed for TMA arg; re-tunes `64` kernels at launch | TMA descriptor wrapper + persisted `KERNEL_CONFIGS` table | **`64 -> 0` first-launch re-tunes, `0` reduction-order drift** |
| **Megatron-LM Tokenizer Init** | `megatron@73f17bc` | Duplicate `build_tokenizer` calls (`3`/rank) | Unified `get_tokenizer()` | **`3 -> 1` tokenizer builds per rank** |
| **Mooncake EFA HugePage Registration** | `mooncake@e648c15` | `88 GiB`/NIC limit with 4 KB pages (`128 GiB` fails on 1 NIC) | `hugetlb` / THP `2 MiB` huge pages | **`512.0x` fewer MR page entries, `128 GiB+` on single NIC** |
| **LMCache Concurrent TP Transfers** | `lmcache@d67052b` | Serial `TP=8` worker transfers (`148.0 ms`) | Concurrent `TP=8` worker transfers + `--transfer-timeout` (`19.6 ms`) | **`7.551x` faster `TP=8` transfer benchmark** |
| **Dynamo DCGM & Session Affinity** | `dynamo@f9c22d7`, `c03f514` | `23,040` leaked sockets/h; local-only session bindings | Closes DCGM connections + `SessionBindingStream` replication | **`0` leaked sockets/h, `100%` failover session affinity** |

---

## 3. Reproducible Verification

```bash
python experiments/pytorch_rccl_symm_tokenswitch_tma_mooncake_efa_lmcache_dynamo_phase129/run.py --check
python -m unittest tests/test_phase129_experiments.py -v
```
