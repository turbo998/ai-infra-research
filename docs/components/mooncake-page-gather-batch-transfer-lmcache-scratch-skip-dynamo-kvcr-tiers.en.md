# Mooncake Page-Gathered Batch RDMA Transfer APIs, LMCache Scratch Group Span-0 Skip, and NVIDIA Dynamo MultiConnector KVCR Discovery

> **中文版**: [mooncake-page-gather-batch-transfer-lmcache-scratch-skip-dynamo-kvcr-tiers.md](mooncake-page-gather-batch-transfer-lmcache-scratch-skip-dynamo-kvcr-tiers.md)

## 1. First-Hand Source Evidence

| System / Repository | Commit / Date | Core Source Paths | Evidence Class |
|---|---|---|---|
| **Mooncake** (`kvcache-ai/Mooncake`) | `387aecb27cc9268be8318c421972ee5e9125a12e` (`2026-10-19`, PR `#4763`) | `mooncake-integration/transfer_engine/transfer_engine_py.cpp`, `transfer_engine_py.h` | `implementation` |
| **LMCache** (`LMCache/LMCache`) | `2a34fa829b3c902c5b6393740239f6ff65bd0ff8` (`2026-10-19`, PR `#5818`) | `lmcache/integration/vllm/lazy_offload_manager.py`, `tests/v1/test_lazy_offload_manager.py` | `implementation` |
| **NVIDIA Dynamo** (`ai-dynamo/dynamo`) | `662cb22df401d8a5285383f83e7defca34cab227` (`2026-10-20`, PR `#16778`) | `components/src/dynamo/vllm/kv_hints.py`, `components/src/dynamo/vllm/tests/test_kv_hints.py` | `implementation` |

---

## 2. Mooncake TransferEngine: Buffer-Protocol (`_arrays`) and Page-Gathered (`_pages`) Batch APIs (`#4763`)

### 2.1 Eliminating pybind11 Element-by-Element List Unboxing in Disaggregated PD Transfers
In Prefill-Decode (PD) disaggregation, transferring a single long prompt's KV cache spans tens of thousands of descriptors ($F$ fields/layers $\times$ $P$ paged KV blocks). With the legacy list-based batch APIs, Python callers had to allocate three `list[int]` objects of length $F \times P$ and let pybind11 unbox $3 \times F \times P$ Python integers one element at a time.

### 2.2 Contiguous Buffer Protocol and C++ Field-Major Cartesian Expansion
Commit `387aecb2` adds two new descriptor ingestion paths in `transfer_engine_py.cpp`:
1. **`batch_transfer_{sync,async}_{write,read}_arrays`**: Reads `buffers`, `peer_buffer_addresses`, and `lengths` directly from 1-D C-contiguous 8-byte integer buffers (`QLqlNn` format codes, validating non-negativity for signed `qln`) via a single `std::memcpy`.
2. **`batch_transfer_sync_{write,read}_pages`**: Accepts 1-D `src_pages[P]`, `dst_pages[P]`, and a C-contiguous `[F, 5]` `fields` matrix `(src_base, src_stride, dst_base, dst_stride, length)`, expanding all $F \times P$ descriptors **field-major** in C++:
   $$\text{src\_addr}(f, i) = \text{src\_base}_f + \text{src\_pages}[i] \times \text{src\_stride}_f, \quad \text{dst\_addr}(f, i) = \text{dst\_base}_f + \text{dst\_pages}[i] \times \text{dst\_stride}_f$$
   and submitting chunks of at most `max_batch_size` to `batchTransferSync`.
- **Quantitative Impact**: For $F=64, P=512$ ($32,768$ descriptors), cross-language integer inputs drop from $98,304$ boxed Python ints to $2 \times 512 + 5 \times 64 = 1,344$ contiguous 8B ints (**$98.6328\%$ reduction**), submitted in $4$ chunks at `max_batch_size=8192`.

---

## 3. LMCache: Skipping Scratch Groups (`tokens_per_block = 0`) in `LazyOffloadManager` (`#5818`)

In hybrid attention/state models (such as Qwen3.8-Flash-Next and GLM-5.3-Flash), a scratch ring group holds a single block per request for its lifetime and reports `tokens_per_block = 0` alongside cacheable KV groups (e.g., `[16, 0, 32]`). Previously, `LazyOffloadManager._drain()` evaluated `-(-total_num_scheduled_tokens // tokens_per_block)` across all groups, raising `ZeroDivisionError` on span `0`.
Commit `2a34fa82` guards the generator with `if tokens_per_block > 0`, yielding $\lceil 48/16 \rceil + \lceil 48/32 \rceil = 5$ blocks for $48$ scheduled tokens on `[16, 0, 32]`.

---

## 4. NVIDIA Dynamo: Discovering `kvcr` Secondary Tiers Inside `MultiConnector` (`#16778`)

When vLLM combines `NixlConnector` and `OffloadingConnector` under `MultiConnector` (`MULTI_CONNECTOR_WRAPPERS`), `secondary_tiers` lives inside the child `OffloadingConnector`'s `kv_connector_extra_config` rather than the top-level `MultiConnector` dictionary. Commit `662cb22d` unwraps `extra_config.get("connectors")` whenever `kv_connector in MULTI_CONNECTOR_WRAPPERS`, enabling `publish_kv_hint_capabilities` to advertise `kv_hint_transfer = true` and control endpoint `{"0": "tcp://127.0.0.1:17771"}`.

---

## 5. Reproducible Offline Verification

```bash
python experiments/mooncake_page_gather_lmcache_scratch_dynamo_kvcr/run.py --check
```
