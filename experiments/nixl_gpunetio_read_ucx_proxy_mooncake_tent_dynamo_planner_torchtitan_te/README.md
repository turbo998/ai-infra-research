# NIXL GPUNetIO Contiguous READ Coalescing & UCX CPU Proxy EP, Mooncake Store `TentRequest` 48-Byte C ABI & Snapshot Btime Sort, Dynamo Planner Warmup Reconfiguration, and TorchTitan / TransformerEngine JAX EP

Deterministic Python 3.12 standard-library verification for **Phase 46 (`priority: 55`)** Topic 3 (`q-phase46-nixl-gpunetio-read-ucx-proxy-mooncake-tent-dynamo-planner-torchtitan-te`).

## First-Hand Upstream Commits
- `nixl@1b84929f8a598aa5aae5945aec7ed817cfeaa513` (`#2176`): GPUNetIO contiguous READ coalescing within and across the 512-input `DOCA_XFER_REQ_SIZE` staging boundary (`8 MiB` cap, `4096 -> 2` WQEs, `8 -> 2` ring entries, `+109.92%` throughput `10.97 -> 23.03 GB/s` and `-52.11%` p99 latency `1516 -> 726 us` on 2× H20).
- `nixl@3f6e1a2716e728b3a9e16b1556a9b51efbebd19b` (`#2327`) & `nixl@fda3a883c0aef00e3b42c4a0f08899aa0b49d3c6` (`#2328`): UCX CPU Proxy (`device_proxy=true`, `threadShare(thread)`) for expert-parallel dispatch/combine (`82.97 / 80.23 GB/s` vs `81.90 / 76.62 GB/s` GPU RDMA on 16× GB200).
- `Mooncake@7e4fc11ee0fe9553a82a408009ea2b499c76842e` (`#4732`) & `Mooncake@1a6d00d0f4dc7f54c573c5fcd4ff5f80bd1f2813` (`#4688`): `mooncake-store-rs` `TentRequest` 48-byte C ABI alignment (`priority` + `transport_hint`) and `FileUtil::SortByCreationTime` `TimeKey` snapshot strict weak ordering.
- `dynamo@a76f1efaed1936abe9c8e6c9acb0897445e8f57c` (`#16843`): Planner live reconfiguration (`/engine/planner`, HTTP 409 revision guard, `16 MiB` / `4096`-interval off-loop predictor warmup, atomic `install_prepared_predictors`).
- `torchtitan@9391c0110aa9705b4bf8a1410c3c847ea6cc8fa1` (`#5389`), `torchtitan@c0eb136092402581bc1b184434bd2e46fea90bbc` (`#5501`), & `TransformerEngine@6ca7b5aa88ef2e1497f7a896ca1d9bd80442222e` (`#3639`): `ChunkedLossWrapper` `head_has_trainable_params` for frozen-decoder `lm_head` LoRA, `_HuggingFaceCursorIterator` `_pending_hf_state`, and TransformerEngine JAX `NVTE_EP_HANDLE_CACHE_SIZE=0` + `ncclEpImportHandle`.

## Usage
```bash
python experiments/nixl_gpunetio_read_ucx_proxy_mooncake_tent_dynamo_planner_torchtitan_te/run.py --check
```
